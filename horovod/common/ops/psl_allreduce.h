#ifndef HOROVOD_PSL_OPERATIONS_H
#define HOROVOD_PSL_OPERATIONS_H

//#include <nccl.h>

#include "cuda_operations.h"
#include "mpi.h"
#include <emmintrin.h>
#include <immintrin.h>

namespace horovod {
namespace common {

typedef uint16_t float16;

template <typename T> class DeviceImpl {
public:
  static void DotProd(const T*, const T*, int n, double&, double&, double&);
  static void ScaleAdd(int, float, T*, float, const T*);
};

template <> class DeviceImpl<float> {
public:
  static void DotProd(const float* __restrict__ a, const float* __restrict__ b,
                      int n, double& dotProduct, double& anormsq,
                      double& bnormsq) {
    dotProduct = 0;
    anormsq = 0;
    bnormsq = 0;
    for (int i = 0; i < n; i++) {
      dotProduct += a[i] * b[i];
      anormsq += a[i] * a[i];
      bnormsq += b[i] * b[i];
    }
  }

  static void ScaleAdd(int n, float acoeff, float* __restrict__ a, float bcoeff,
                       const float* __restrict__ b) {
    for (int i = 0; i < n; i++) {
      a[i] = acoeff * a[i] + bcoeff * b[i];
    }
  }
};

template <> class DeviceImpl<float16> {
public:
  static void DotProd(const float16* a, const float16* b, int len,
                      double& dotProduct, double& anormsq, double& bnormsq) {
    int i;
    __m256d dotProductVec = _mm256_setzero_pd();
    __m256d anormVec = _mm256_setzero_pd();
    __m256d bnormVec = _mm256_setzero_pd();
    for (i = 0; i < len - 7; i += 8) {
      __m256 aVec = _mm_loadu_ph(&a[i]);
      __m256 bVec = _mm_loadu_ph(&b[i]);
      __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
      __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
      __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
      __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
      dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
      dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
      anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
      anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
      bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
      bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
    }
    if (i < len) {
      __m256 aVec = _mm_loadu_ph_partial(&a[i], len - i);
      __m256 bVec = _mm_loadu_ph_partial(&b[i], len - i);
      __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
      __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
      __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
      __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
      dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
      dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
      anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
      anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
      bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
      bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
    }

    dotProduct = _mm256Reduction_pd(dotProductVec);
    anormsq = _mm256Reduction_pd(anormVec);
    bnormsq = _mm256Reduction_pd(bnormVec);
  }

  static void ScaleAdd(int len, float acoeff, float16* a, float bcoeff,
                       const float16* b) {
    int i;
    __m256 acoeffVec = _mm256_set1_ps((float)(acoeff));
    __m256 bcoeffVec = _mm256_set1_ps((float)bcoeff);
    for (i = 0; i < len - 7; i += 8) {
      __m256 aVec = _mm_loadu_ph(&a[i]);
      __m256 bVec = _mm_loadu_ph(&b[i]);
      aVec = _mm256_mul_ps(acoeffVec, aVec);
      _mm_store_ph(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec));
    }
    if (i < len) {
      __m256 aVec = _mm_loadu_ph_partial(&a[i], len - i);
      __m256 bVec = _mm_loadu_ph_partial(&b[i], len - i);
      aVec = _mm256_mul_ps(acoeffVec, aVec);
      _mm_store_ph_partial(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec),
                           len - i);
    }
  }

private:
  // reduces 8xfloat32 into one scalar
  inline float static _mm256Reduction(__m256 x) {
    const __m128 x128 =
        _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
  }

  // reduce 4xfloat64 into one double
  inline double static _mm256Reduction_pd(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
  }

  // load 8 float16s from a and return the __m256 register
  inline __m256 static _mm_loadu_ph(const uint16_t* a) {
    __m128i r = _mm_loadu_si128((__m128i*)(a));
    return _mm256_cvtph_ps(r);
  }

  // store 8 float16 from val into a
  inline void static _mm_store_ph(uint16_t* a, __m256 val) {
    __m128i r = _mm256_cvtps_ph(val, 0);
    _mm_storeu_si128((__m128i*)a, r);
  }

  // load len (< 8) float16s from a, fill the rest with 0s, and return the
  // __m256 register
  inline __m256 static _mm_loadu_ph_partial(const uint16_t* a, int len) {
    short e[8];
    std::memset(e, 0, sizeof(e));
    std::memcpy(e, a, std::min(len, 8) * sizeof(short));
    __m128i es = _mm_set_epi16(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
    return _mm256_cvtph_ps(es);
  }

  // store the first len (< 8) float16s from val and store into a
  inline void static _mm_store_ph_partial(uint16_t* a, __m256 val, int len) {
    __m128i r = _mm256_cvtps_ph(val, 0);
    // for (int i = 0; i < std::min(len, 8); i++)
    //    a[i].value = _mm_extract_epi16(r, i);
    // but we cannot do this because the second argument to _mm_extract_epi16
    // has to be a compile time constant
    if (0 < len)
      a[0] = (short)_mm_extract_epi16(r, 0);
    if (1 < len)
      a[1] = (short)_mm_extract_epi16(r, 1);
    if (2 < len)
      a[2] = (short)_mm_extract_epi16(r, 2);
    if (3 < len)
      a[3] = (short)_mm_extract_epi16(r, 3);
    if (4 < len)
      a[4] = (short)_mm_extract_epi16(r, 4);
    if (5 < len)
      a[5] = (short)_mm_extract_epi16(r, 5);
    if (6 < len)
      a[6] = (short)_mm_extract_epi16(r, 6);
    if (7 < len)
      a[7] = (short)_mm_extract_epi16(r, 7);
  }
};

template <typename T>
static void AllreduceImplTree(T* grad_buffer, T* recv_buffer, int count,
                              MPI_Comm communicator, int message_tag) {
  int true_rank;
  int redn_rank;
  int size;
  MPI_Comm_rank(communicator, &true_rank);
  MPI_Comm_size(communicator, &size);

  static int counter = 0;
  int root_node_rotation = false ? (counter++ % size) : 0;
  redn_rank = (true_rank ^ root_node_rotation);

  // Do a tree reduction
  // The reduction ranks used are a permutation of true ranks (permuted based on
  // message_tag) This spreads the load of tree reduction across different true
  // ranks

  // at each level l, node X0[0..0] receives from X1[0...],
  // where [0..0] is l zeros in the bit representation of the rank of a node
  int level;
  for (level = 1; level < size; level *= 2) {
    int neighbor_redn_rank = redn_rank ^ level;
    int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
    if (redn_rank % level != 0)
      continue; // stay idle at this level

    if (neighbor_redn_rank >= size)
      continue; // no neighbor and so stay idle at this level

    if ((redn_rank & level) == 0) {
      // recv buffer from neighbor
      MPI_Recv(recv_buffer, count * sizeof(T), MPI_CHAR, neighbor_true_rank,
               message_tag, communicator, MPI_STATUS_IGNORE);
      float acoeff = 1;
      float bcoeff = 1;
      DeviceImpl<T>::ScaleAdd(count, acoeff, grad_buffer, bcoeff, recv_buffer);
    } else {
      // send grad_buffer to neighbor
      MPI_Send(grad_buffer, count * sizeof(T), MPI_CHAR, neighbor_true_rank,
               message_tag, communicator);
    }
  }

  // the same logic as above.
  // at each level l, node X0[0..0] sends to X1[0...],
  // where [0..0] is l zeros in the bit representation of the rank of a node

  level /= 2; // this make sure that level < size

  for (; level > 0; level /= 2) {
    int neighbor_redn_rank = redn_rank ^ level;
    int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);

    if (redn_rank % level != 0)
      continue; // stay idle at this level

    if (neighbor_redn_rank >= size)
      continue; // no neighbor and so stay idle at this level

    if ((redn_rank & level) == 0) {
      // send grad_buffer to neighbor
      // and dont wait for the send to finish
      MPI_Send(grad_buffer, count * sizeof(T), MPI_CHAR, neighbor_true_rank,
	       message_tag, communicator);
    }
    else {
      // recv grad_buffer from neighbor
      MPI_Recv(grad_buffer, count * sizeof(T), MPI_CHAR, neighbor_true_rank, message_tag,
	       communicator, MPI_STATUS_IGNORE);
    }
  }
}

static std::vector<double> normAndDots;

template <typename T>
static void PairwiseReduceWithComm(T* a, T* b, std::vector<int>& tensor_counts, int layerid,
                                   MPI_Comm& comm, bool isLeftNeighbor) {
  normAndDots.resize(tensor_counts.size()*3);

  int countSoFar = 0;
  for (int i = 0; i < tensor_counts.size(); i++){
    double dotProduct = 0.;
    double anormsq = 0.;
    double bnormsq = 0.;

    DeviceImpl<T>::DotProd(&a[countSoFar], &b[countSoFar], tensor_counts[i], dotProduct, anormsq, bnormsq);
    normAndDots[i*3] = dotProduct;
    if (isLeftNeighbor) {
      normAndDots[i*3+1] = anormsq;
      normAndDots[i*3+2] = bnormsq;
    } else {
      normAndDots[i*3+1] = bnormsq;
      normAndDots[i*3+2] = anormsq;
    }
    countSoFar += tensor_counts[i];
  }
  // double reduce_vals[3];
  // if (isLeftNeighbor) {
  //   reduce_vals[0] = anormsq;
  //   reduce_vals[1] = bnormsq;
  // } else {
  //   reduce_vals[1] = anormsq;
  //   reduce_vals[0] = bnormsq;
  // }
  // reduce_vals[2] = dotProduct;

  // TODO replace this with something else
  // AllreduceImplTree(reduce_vals, temp_reduce_vals, 3, comm, layerid);
  MPI_Allreduce(MPI_IN_PLACE, normAndDots.data(), 3*tensor_counts.size(), MPI_DOUBLE, MPI_SUM, comm);

  countSoFar = 0;
  for (int i = 0; i < tensor_counts.size(); i++){
    double dotProduct = normAndDots[i*3];
    double anormsq;
    double bnormsq;
    if (isLeftNeighbor) {
      anormsq = normAndDots[i*3+1];
      bnormsq = normAndDots[i*3+2];
    } else {
      bnormsq = normAndDots[i*3+1];
      anormsq = normAndDots[i*3+2];
    }

    double acoeff = 1;
    double bcoeff = 1;
    if (anormsq >= 1e-8)
      acoeff = 1.0 - dotProduct / anormsq * 0.5;
    if (bnormsq >= 1e-8)
      bcoeff = 1.0 - dotProduct / bnormsq * 0.5;

    DeviceImpl<T>::ScaleAdd(tensor_counts[i], acoeff, &a[countSoFar], bcoeff, &b[countSoFar]);
    countSoFar += tensor_counts[i];
  }

  // if (isLeftNeighbor) {
  //   anormsq = reduce_vals[0];
  //   bnormsq = reduce_vals[1];
  // } else {
  //   anormsq = reduce_vals[1];
  //   bnormsq = reduce_vals[0];
  // }
  // dotProduct = reduce_vals[2];

  // double acoeff = 1;
  // double bcoeff = 1;
  // if (anormsq >= 1e-8)
  //   acoeff = 1.0 - dotProduct / anormsq * 0.5;
  // if (bnormsq >= 1e-8)
  //   bcoeff = 1.0 - dotProduct / bnormsq * 0.5;

  // a = acoeff * a + bcoeff * b
}

static bool IsPowerOfTwo(ulong x) { return (x != 0) && ((x & (x - 1)) == 0); }
static std::vector<std::vector<int>> nghrCountVec;

template <typename T>
static void AllreduceImplVHDD(T* grad_buffer, T* recv_buffer, std::vector<int>& tensor_counts,
                              int start_level, MPI_Comm communicator,
                              int layerid, MPI_Comm* reduction_comms) {
  int rank;
  int size;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

	printf("%d Here 4\n\n", rank); std::fflush(stdout);
  if (IsPowerOfTwo(size) == false) {
    throw std::logic_error(
        "BUGBUG: need to implement logic for non power of two ranks");
  }

  // std::cerr << " rere0 " << rank << std::endl;
  // int chunk_size = (1<<15);
  int chunk_size = (1 << 29);
  int nearest_power_2 = 1;
  for (nearest_power_2 = 1; (nearest_power_2 << 1) <= size;
       nearest_power_2 = (nearest_power_2 << 1)) {
  }
  int remaining_non_power_2 = size - nearest_power_2;
  int level;
	printf("%d Here 5\n\n", rank); std::fflush(stdout);
  if (rank >= size - 2 * remaining_non_power_2) {
    printf("Allreduce is not implemented for non-power2 number of ranks\n");
    exit(-1);
    // int myCount;
    // int nghrCount;
    // level = 0;
    // int neighbor_rank;
    // int sendOffset;
    // int recvOffset;
    // if (rank < nearest_power_2) {
    //   neighbor_rank = rank + remaining_non_power_2;
    //   myCount = (count >> 1);
    //   nghrCount = count - myCount;
    //   sendOffset = myCount;
    //   recvOffset = 0;
    // } else {
    //   nghrCount = (count >> 1);
    //   myCount = count - nghrCount;
    //   neighbor_rank = rank - remaining_non_power_2;
    //   sendOffset = 0;
    //   recvOffset = nghrCount;
    // }
    // for (int i = 0; i < std::max(nghrCount, myCount); i += chunk_size) {
    //   MPI_Sendrecv(
    //       (char*)(&grad_buffer[i + sendOffset]),
    //       std::min(chunk_size, nghrCount - i) * sizeof(T) / sizeof(char),
    //       MPI_CHAR, neighbor_rank, layerid,
    //       (char*)(&recv_buffer[i + recvOffset]),
    //       std::min(chunk_size, myCount - i) * sizeof(T) / sizeof(char),
    //       MPI_CHAR, neighbor_rank, layerid, communicator, MPI_STATUS_IGNORE);
    // }

    // DeviceImpl<T>::ScaleAdd(myCount, 1.0, &grad_buffer[recvOffset] , 1.0, &recv_buffer[recvOffset]);

    // if (rank < nearest_power_2) {
    //   for (int i = 0; i < nghrCount; i += chunk_size) {
    //     MPI_Recv((char*)(&grad_buffer[i + sendOffset]),
    //              std::min(chunk_size, nghrCount - i) * sizeof(T) / sizeof(char),
    //              MPI_CHAR, neighbor_rank, layerid, communicator,
    //              MPI_STATUS_IGNORE);
    //   }
    // } else {
    //   for (int i = 0; i < myCount; i += chunk_size)
    //     MPI_Send((char*)(&grad_buffer[i + recvOffset]),
    //              std::min(chunk_size, myCount - i) * sizeof(T) / sizeof(char),
    //              MPI_CHAR, neighbor_rank, layerid, communicator);
    // }
  }

  int nghrCountVec_index = 0;
  int orgSize = size;
  size = nearest_power_2;
	printf("%d Here 6\n\n", rank); std::fflush(stdout);
  if (rank < nearest_power_2) {

    int total_counts_sum = 0;
    for (int i = 0; i < tensor_counts.size(); i++)
      total_counts_sum += tensor_counts[i];
    int myCount = total_counts_sum;
    int comm_index;
    for (level = 1, comm_index = 0; level < size;
         level = (level << 1), comm_index++) {
      if (level < start_level) {
        continue;
      }

      int neighbor_rank = rank ^ level;
      int nghrCount = 0;
      int sendOffset = 0;
      int recvOffset = 0;
      int firstHalfMyCount = (myCount >> 1);
      int secondHalfMyCount = myCount - firstHalfMyCount;

      if (nghrCountVec.size() <= nghrCountVec_index)
        nghrCountVec.emplace_back();
			nghrCountVec[nghrCountVec_index].resize(tensor_counts.size());
			printf("%d Here 7\n\n", rank); std::fflush(stdout);

      int myCountSoFar = 0;
      int nghrCountSoFar = 0;
      if ((rank & level) != 0) {
        myCount = secondHalfMyCount;
        nghrCount = firstHalfMyCount;
        sendOffset = 0;
        recvOffset = nghrCount;

				printf("%d Here 8\n\n", rank); std::fflush(stdout);
        for (int i = 0; i < tensor_counts.size(); i++){
          if (nghrCountSoFar <= nghrCount){
            if (nghrCountSoFar+tensor_counts[i] <= nghrCount){
              nghrCountVec[nghrCountVec_index][i] = tensor_counts[i];
              tensor_counts[i] = 0;
            } else {
              nghrCountVec[nghrCountVec_index][i] = nghrCount - nghrCountSoFar; // should not be negative
              tensor_counts[i] = tensor_counts[i] - (nghrCount - nghrCountSoFar); // should not be negative
            }
          } else {
            tensor_counts[i] = tensor_counts[i];
            nghrCountVec[nghrCountVec_index][i] = 0;
          }
          nghrCountSoFar += nghrCountVec[nghrCountVec_index][i];
          myCountSoFar += tensor_counts[i];
        }
				printf("%d Here 9\n\n", rank); std::fflush(stdout);
      } else {
        myCount = firstHalfMyCount;
        nghrCount = secondHalfMyCount;
        sendOffset = myCount;
        recvOffset = 0;

				printf("%d Here 10\n", rank); std::fflush(stdout);
        for (int i = 0; i < tensor_counts.size(); i++){
          if (myCountSoFar <= myCount){
            if (myCountSoFar+tensor_counts[i] <= myCount){
              tensor_counts[i] = tensor_counts[i];
              nghrCountVec[nghrCountVec_index][i] = 0;
            } else {
              nghrCountVec[nghrCountVec_index][i] = tensor_counts[i] - (myCount - myCountSoFar); // should not be negative
              tensor_counts[i] = myCount - myCountSoFar; // should not be negative
            }
          } else {
            nghrCountVec[nghrCountVec_index][i] = tensor_counts[i];
            tensor_counts[i] = 0;
          }
          nghrCountSoFar += nghrCountVec[nghrCountVec_index][i];
          myCountSoFar += tensor_counts[i];
        }
				printf("%d Here 11\n", rank); std::fflush(stdout);
      }

			int test_count = 0;
			int total_myCount = 0;
			int total_nghrCount = 0;
			for (int i = 0; i < tensor_counts.size(); i++){
				if (tensor_counts[i] < 0 || nghrCountVec[nghrCountVec_index][i] < 0){
					printf("QQQQQQ something went wrong!\n");
					exit(-1);
				}
				total_myCount += tensor_counts[i];
				total_nghrCount += nghrCountVec[nghrCountVec_index][i];
				test_count += tensor_counts[i] + nghrCountVec[nghrCountVec_index][i];
			}
			if (total_myCount != myCount || total_nghrCount != nghrCount){
				printf("SSSSSSSSSS something went wrong!\n");
				exit(-1);
			}

			if (test_count != total_counts_sum){
				printf("TTTTTTTTTT something went wrong!\n");
				exit(-1);
			}



      if (nghrCountSoFar != nghrCount || myCountSoFar != myCount){
        printf("!!!!!!!!!!! Something went wrong!\n");
        exit(-1);
      }
      
      nghrCountVec_index++;
			
			for (int i = 0; i < tensor_counts.size(); i++){
				printf("rank %d total_counts_sum %d sendOffset %d recvOffset %d myCount %d nghrCount %d\n", rank, total_counts_sum, sendOffset, recvOffset, myCount, nghrCount);
			}

			printf("%d Here 12\n", rank); std::fflush(stdout);
      for (int i = 0; i < std::max(myCount, nghrCount); i += chunk_size)
        MPI_Sendrecv(
            (char*)(&grad_buffer[i + sendOffset]),
            std::min(chunk_size, nghrCount - i) * sizeof(T) / sizeof(char),
            MPI_CHAR, neighbor_rank, layerid,
            (char*)(&recv_buffer[i + recvOffset]),
            std::min(chunk_size, myCount - i) * sizeof(T) / sizeof(char),
            MPI_CHAR, neighbor_rank, layerid, communicator, MPI_STATUS_IGNORE);
      if ((rank & level) != 0) {
        grad_buffer = &grad_buffer[nghrCount];
        recv_buffer = &recv_buffer[nghrCount];
      }
			printf("%d Here 13\n", rank); std::fflush(stdout);
      PairwiseReduceWithComm<T>(grad_buffer, recv_buffer, tensor_counts, layerid,
                                reduction_comms[comm_index],
                                (rank & level) == 0);
			printf("%d Here 14\n", rank); std::fflush(stdout);
			return;
    }

    for (level = (size >> 1); level > 0; level = (level >> 1)) {
			printf("%d Here 15\n", rank); std::fflush(stdout);
      if (level < start_level){
        continue;
      }
      int neighbor_rank = rank ^ level;

			printf("%d Here 16\n", rank); std::fflush(stdout);
      nghrCountVec_index--;
      int nghrCount = 0;
      for (int i = 0; i < tensor_counts.size(); i++){
        nghrCount += nghrCountVec[nghrCountVec_index][i];
        tensor_counts[i] += nghrCountVec[nghrCountVec_index][i];
      }
			printf("%d Here 17\n", rank); std::fflush(stdout);

      if ((rank & level) == 0) {
        recv_buffer = &grad_buffer[myCount];
      } else {
        recv_buffer = &grad_buffer[-nghrCount];
      }
			printf("%d Here 18\n", rank); std::fflush(stdout);
      for (int i = 0; i < std::max(myCount, nghrCount); i += chunk_size)
        MPI_Sendrecv(
            (char*)(&grad_buffer[i]),
            std::min(chunk_size, myCount - i) * sizeof(T) / sizeof(char),
            MPI_CHAR, neighbor_rank, layerid, (char*)(&recv_buffer[i]),
            std::min(chunk_size, nghrCount - i) * sizeof(T) / sizeof(char),
            MPI_CHAR, neighbor_rank, layerid, communicator, MPI_STATUS_IGNORE);
			printf("%d Here 19\n", rank); std::fflush(stdout);
      if ((rank & level) != 0) {
        grad_buffer = &grad_buffer[-nghrCount];
      }
      myCount += nghrCount;
    }
		printf("%d Here 20\n", rank); std::fflush(stdout);
  }
  size = orgSize;

  if (rank >= size - 2 * remaining_non_power_2) {
    // level = 0;
    // int neighbor_rank;
    // if (rank < nearest_power_2) {
    //   neighbor_rank = rank + remaining_non_power_2;
    //   for (int i = 0; i < count; i += chunk_size) {
    //     MPI_Send((char*)(&grad_buffer[i]),
    //              std::min(chunk_size, count - i) * sizeof(T) / sizeof(char),
    //              MPI_CHAR, neighbor_rank, layerid, communicator);
    //   }
    // } else {
    //   neighbor_rank = rank - remaining_non_power_2;
    //   for (int i = 0; i < count; i += chunk_size)
    //     MPI_Recv((char*)(&grad_buffer[i]),
    //              std::min(chunk_size, count - i) * sizeof(T) / sizeof(char),
    //              MPI_CHAR, neighbor_rank, layerid, communicator,
    //              MPI_STATUS_IGNORE);
    // }
  }
  // std::cerr << " rere1 " << rank << std::endl;
}

static MPI_Comm* reduction_comms = NULL;

template <typename T>
static void AllreduceImpl(T* grad_buffer, T* recv_buffer, std::vector<int>& tensor_counts, int start_level,
                          MPI_Comm communicator, int layerid) {
  //   if (count <= 1024)
  //     AllreduceImplTree<T>(grad_buffer, recv_buffer, count, communicator,
  //                          layerid);
  //   else {
  //     if (reduction_comms == NULL) {
  //       throw std::logic_error("init comms please");
  //     }
  AllreduceImplVHDD<T>(grad_buffer, recv_buffer, tensor_counts, start_level, communicator, layerid,
                       reduction_comms);
  //   }
}

static void InitComms(MPI_Comm global_comm) {
  int rank, size;
  MPI_Comm_rank(global_comm, &rank);
  MPI_Comm_size(global_comm, &size);

  MPI_Group world_group;
  MPI_Comm_group(global_comm, &world_group);

  int nearest_power_2 = 1;
  int log_size;
  for (nearest_power_2 = 1, log_size = 0; (nearest_power_2 << 1) <= size;
       nearest_power_2 = (nearest_power_2 << 1), log_size++)
    ;
  int shift_val;
  int level;
  int rank_log_size = log_size;
  reduction_comms = new MPI_Comm[log_size];
  int* node_rank = new int[size];
  for (level = 1, shift_val = 1; level < nearest_power_2;
       level = (level << 1), shift_val++) {
    int base_rank = ((rank >> shift_val) << shift_val);
    for (int i = 0; i < (level << 1); i++) {
      node_rank[i] = (base_rank + i); // * ms_local_size;
    }
    MPI_Group red_group;
    MPI_Group_incl(world_group, (level << 1), node_rank, &red_group);
    MPI_Comm_create_group(global_comm, red_group, 0,
                          &reduction_comms[shift_val - 1]);
    MPI_Group_free(&red_group);
  }

  delete[] node_rank;
}

int PSL_Allreduce(const void* sendbuf, void* recvbuf, std::vector<int>& tensor_counts, int start_level,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  if (op != MPI_OP_NULL) {
    std::cerr << "MPI_op op must be null for parasail logic " << datatype
              << std::endl;
    return MPI_ERR_OP;
  }

  int count = 0;
  for (int i = 0; i < tensor_counts.size(); i++)
    count += tensor_counts[i];

	int rank;
	MPI_Comm_rank(comm, &rank);
  printf("%d Here 0\n\n", rank); std::fflush(stdout);
  if (reduction_comms == NULL) {
    std::cerr << "initializing comms" << std::endl;
    InitComms(comm);
  }
  printf("%d Here 1\n\n", rank); std::fflush(stdout);

  static std::vector<char> tmp;
  const void* input_buffer;
  void* tmp_buffer;

  if (datatype == MPI_FLOAT) {
    if (sendbuf == MPI_IN_PLACE) {
      tmp.resize(sizeof(float) * count);
      tmp_buffer = tmp.data();
      input_buffer = recvbuf;
    } else {
      input_buffer = sendbuf;
      tmp_buffer = recvbuf;
      throw std::logic_error("not implemented");
    }
		printf("%d Here 2\n\n", rank); std::fflush(stdout);
    AllreduceImpl<float>((float*)input_buffer, (float*)tmp_buffer, tensor_counts, start_level, comm, 0);
		printf("%d Here 3\n\n", rank); std::fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
		printf("%d Here 30\n", rank); std::fflush(stdout);
    return MPI_SUCCESS;
  } else if (datatype == MPI_DOUBLE) {
    std::cerr << "not implemented yet " << datatype << std::endl;
    return MPI_ERR_TYPE;
  }
	printf("%d Here 31\n", rank); std::fflush(stdout);

  int size;
  MPI_Type_size(datatype, &size);
  if (size == 2) {
    if (sendbuf == MPI_IN_PLACE) {
      tmp.resize(sizeof(float16) * count);
      tmp_buffer = tmp.data();
      input_buffer = recvbuf;
    } else {
      input_buffer = sendbuf;
      tmp_buffer = recvbuf;
      throw std::logic_error("not implemented");
    }
    AllreduceImpl<float16>((float16*)input_buffer, (float16*)tmp_buffer, tensor_counts, start_level, comm, 0);
    return MPI_SUCCESS;
  } else {
    std::cerr << "unknown MPI type " << datatype << " size " << size << std::endl;
    return MPI_ERR_TYPE;
  }
  return MPI_SUCCESS;
}

} // namespace common
} // namespace horovod

#endif // HOROVOD_PSL_OPERATIONS_H
