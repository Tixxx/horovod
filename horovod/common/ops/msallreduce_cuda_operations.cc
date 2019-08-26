// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Microsoft Corp.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "msallreduce_cuda_operations.h"
#include "msallreduce_cuda_kernels.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {

/*
std::string CublasContext::GetCublasErrorString (cublasStatus_t cublas_result) {
  switch (cublas_result) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return std::string("CUBLAS_STATUS_NOT_INITIALIZED");
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        return std::string("CUBLAS_STATUS_ALLOC_FAILED");
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        return std::string("CUBLAS_STATUS_INVALID_VALUE");
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return std::string("CUBLAS_STATUS_ARCH_MISMATCH");
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        return std::string("CUBLAS_STATUS_MAPPING_ERROR");
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return std::string("CUBLAS_STATUS_EXECUTION_FAILED");
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return std::string("CUBLAS_STATUS_INTERNAL_ERROR");
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return std::string("CUBLAS_STATUS_NOT_SUPPORTED");
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        return std::string("CUBLAS_STATUS_LICENSE_ERROR");
        break;
      default:
        return std::string("Unknown CUBLAS error!");
  }
}

void CublasContext::ErrorCheck(std::string op_name, cublasStatus_t cublas_result) {
    if (cublas_result != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(std::string(op_name) + " failed: " + GetCublasErrorString(cublas_result));
    }
}
*/

template<typename T>
cudaDataType_t CublasContext::GetCublasDataType(T* variable) {
  if(typeid(T) == typeid(uint16_t)){
      return CUDA_R_16F;
  }
  if(typeid(T) == typeid(float)) {
      return CUDA_R_32F;
  }
  if(typeid(T) == typeid(double)) {
      return CUDA_R_64F;
  }
  throw std::logic_error("Unsupported CUDA type!");
}

//thread_local cublasHandle_t MsCudaAllreduceOp::cublas_Handle;
thread_local double* MsCudaAllreduceOp::device_normsq_memory_a;
thread_local double* MsCudaAllreduceOp::device_normsq_memory_b;
thread_local double* MsCudaAllreduceOp::device_dot_product_memory;

MsCudaAllreduceOp::MsCudaAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : MsAllreduceOp(mpi_context, global_state), cuda_context_(cuda_context) {
    }

void MsCudaAllreduceOp::InitCUDA(const TensorTableEntry& entry, int layerid) {
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));

  LOG(INFO, global_state_->rank)<<"Checking for existing stream for layer "<<layerid<<" "<<std::this_thread::get_id();
  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[global_state_->current_nccl_stream][layerid % global_state_->num_msallreduce_threads];
  if (stream == nullptr) {

    std::lock_guard<std::mutex> guard(global_state_->mutex);
    if (stream == nullptr) {
      LOG(INFO, global_state_->rank)<<"Stream is null, creating new stream "<<std::this_thread::get_id();
      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
    }
  }
  cudaStream_t& device_stream = cuda_context_->streams[global_state_->current_nccl_stream][entry.device];
  if (device_stream == nullptr) {
    std::lock_guard<std::mutex> guard(global_state_->mutex);
    if (stream == nullptr) {
      LOG(INFO, global_state_->rank)<<"device Stream is null, creating new device stream "<<std::this_thread::get_id();
      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&device_stream, cudaStreamNonBlocking, greatest_priority));
    }
  }
  cuda_context_->ErrorCheck("cudaMalloc",
                            cudaMalloc(&device_normsq_memory_a, sizeof(double)));
  cuda_context_->ErrorCheck("cudaMalloc",
                            cudaMalloc(&device_normsq_memory_b, sizeof(double)));
  cuda_context_->ErrorCheck("cudaMalloc",
                            cudaMalloc(&device_dot_product_memory, sizeof(double)));

//  auto status = cublasCreate(&cublas_Handle);
//  CublasContext::ErrorCheck("cublasCreate", status);

//  cublasSetStream(cublas_Handle, stream);
//  cudaStreamSynchronize(stream);
//  LOG(INFO, global_state_->rank)<<"Successfully initialized cublas. "<<std::this_thread::get_id();
}

void MsCudaAllreduceOp::FinalizeCUDA() {
		if (device_normsq_memory_a != nullptr){
			cudaFree(device_normsq_memory_a);
			cudaFree(device_normsq_memory_b);
			cudaFree(device_dot_product_memory);
		}
/*    if(cublas_Handle != nullptr) {
      auto status = cublasDestroy(cublas_Handle);
      CublasContext::ErrorCheck("cublasDestroy", status);
    }*/
}

Status MsCudaAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
      if(entries.size() < 1) {
      return Status::OK();
  }
  //TODO how do we report statuses?
  std::map<int, Status> return_statuses;
  int layerid = 0;
  int num_reductions = entries.size();
  LOG(INFO, global_state_->rank)<<"Ready to process "<<num_reductions<<" tensors in gpu";
  global_state_->finished_parallel_reductions = 0;
  for (auto& entry : entries) {
    boost::asio::post(*global_state_->background_thread_pool,
    [&return_statuses, this, &entry, response, layerid, &entries]
    {
      void* buffer_data;
      int buffer_len;
      void* recv_buffer;

      buffer_data = (void*) entry.tensor->data();

      buffer_len = entry.output->size();

      FusionBufferManager buffer_manager;

      if(entry.tensor->data() == entry.output->data()) {
          LOG(INFO, global_state_->rank)<<"Output and input pointing to same data. Creating temp buffer "<<std::this_thread::get_id();

          // Get the temp buffer to be used for the Op
          global_state_->buffer_lock.lock();
          assert(!global_state_->temp_buffers.empty());
          buffer_manager = global_state_->temp_buffers.front();
          global_state_->temp_buffers.pop();
          global_state_->buffer_lock.unlock();

          // TODO: Maybe add before and after callbacks to timeline?
          Status status = buffer_manager.InitializeBuffer(
              buffer_len,
              entry.device, entry.context,
              global_state_->current_nccl_stream,
              [](){},
              [](){},
              [](int64_t& size, int64_t& threshold){return size >= threshold;});

          if (!status.ok()) {
              throw std::logic_error("MsAllreduceOp::Execute_helper: Initialize buffer failed.");
              return;
          }
          auto& buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
          recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
      }
      else {
          recv_buffer = (void*) entry.output->data();
      }
      LOG(INFO, global_state_->rank)<<"Begin to process gpu tensor with size "<<entry.tensor->size()<<" into output buffer with size "<<entry.output->size()<<" "<<std::this_thread::get_id();
      
      MPI_Comm* node_comm = NULL;
      if (global_state_->rank_log_size != 0) {
          node_comm = &global_state_->reduction_comms[global_state_->rank_log_size-1];
      }
    
      // This will create a stream per layer.
      InitCUDA(entry, layerid);
      LOG(INFO, global_state_->rank)<<"Begin processing gpu tensor in layer "<<layerid<<" "<<std::this_thread::get_id();
      switch (entry.output->dtype()) {
          case HOROVOD_FLOAT16:
            MsAllreduce_Internal((uint16_t*) buffer_data,
                            (uint16_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<uint16_t>,
                            ScaleAddImpl<uint16_t>);  
          break;
          case HOROVOD_FLOAT32:
            MsAllreduce_Internal((float*) buffer_data,
                            (float*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<float>,
                            ScaleAddImpl<float>);  
          break;
          case HOROVOD_FLOAT64:
            MsAllreduce_Internal((double*) buffer_data,
                            (double*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<double>,
                            ScaleAddImpl<double>);  
          
          break;
          default:
            throw std::logic_error("MsAllreduceOp::Execute: Unsupported data type.");
      }
      LOG(INFO, global_state_->rank)<<"Done processing tensor in layer "<<layerid;
      if(entry.tensor->data() == entry.output->data()) {
        // Return the buffer back into the pool of available buffers
        global_state_->buffer_lock.lock();
        global_state_->temp_buffers.push(buffer_manager);
        global_state_->buffer_lock.unlock();
      }
      else {
        memcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size(), layerid);
      }
      LOG(INFO, global_state_->rank)<<"Finished ms gpu allreduction, exiting operation";
      FinalizeCUDA();
      global_state_->finished_parallel_reductions++;
    });
    layerid++;
  }
  while (global_state_->finished_parallel_reductions < num_reductions) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(50));
  }
  return Status::OK();

}

void MsCudaAllreduceOp::memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) {
    assert(dest != nullptr);
    assert(src != nullptr);
    LOG(INFO, global_state_->rank)<<"memcpyUtil GPU. "<<std::this_thread::get_id()<<" for entry device "<<entry.device;
    auto cuda_result = cudaMemcpyAsync(dest, src,
                                    buffer_len, 
                                    cudaMemcpyDeviceToDevice,
                                    cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    auto cuda_sync_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_sync_result);
}

template<typename T>
void MsCudaAllreduceOp::DotProductImpl(const T* __restrict__  a, 
                                       const T* __restrict__ b, 
                                       int n, 
                                       double& dotProduct, 
                                       double& anormsq, 
                                       double& bnormsq, 
                                       HorovodGlobalState *global_state,
                                       int layerid) {
  CudaDotProductImpl(n, a, b, device_normsq_memory_a, device_normsq_memory_b, device_dot_product_memory, anormsq, bnormsq, dotProduct);
}

template<typename T>
void MsCudaAllreduceOp::ScaleAddImpl(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state, int layerid) {
  CudaScaleAddImpl(n, a, b, acoeff, bcoeff);
}

/*
template<typename T>
void MsCudaAllreduceOp::DotProductImpl(const T* __restrict__  a, 
                                       const T* __restrict__ b, 
                                       int n, 
                                       double& dotProduct, 
                                       double& anormsq, 
                                       double& bnormsq, 
                                       HorovodGlobalState *global_state,
                                       int layerid) {
  uint8_t* typed_dotProduct = new uint8_t(sizeof(T));
  uint8_t* typed_anormsq = new uint8_t(sizeof(T));
  uint8_t* typed_bnormsq = new uint8_t(sizeof(T));
  cudaDataType_t cuda_type = CublasContext::GetCublasDataType(a);
  auto isFloat16 = cuda_type == CUDA_R_16F;
  cudaDataType_t execution_type = isFloat16 ? CUDA_R_32F : cuda_type;

  LOG(INFO, global_state->rank)<<"computing a dot b";
  auto adotbstatus = cublasDotEx(cublas_Handle, n, (void *)a, cuda_type, 1, (void *)b, cuda_type, 1, (void *)typed_dotProduct, cuda_type, execution_type);
  CublasContext::ErrorCheck("a cublasdot b", adotbstatus);

  LOG(INFO, global_state->rank)<<"computing a dot a";
  auto adotastatus = cublasDotEx(cublas_Handle, n, (void *)a, cuda_type, 1, (void *)a, cuda_type, 1, (void *)typed_anormsq, cuda_type, execution_type);
  CublasContext::ErrorCheck("a cublasdot a", adotastatus);

  LOG(INFO, global_state->rank)<<"computing b dot b";
  auto bdotbstatus = cublasDotEx(cublas_Handle, n, (void *)b, cuda_type, 1, (void *)b, cuda_type, 1, (void *)typed_bnormsq, cuda_type, execution_type);
  CublasContext::ErrorCheck("b cublasdot b", bdotbstatus);
  cudaStream_t stream;
  cublasGetStream(cublas_Handle, &stream);
  auto cuda_sync_result = cudaStreamSynchronize(stream);
  CUDAContext::ErrorCheck("cudaStreamSynchronize", cuda_sync_result);

  //TODO the cast here is not helpful, this is just to keep the function signature consistent with CPU implementation.
  //If overflow happens, we have lost information already. Find a better way to control output type of the cublas calls.
  if(isFloat16 == true) {
    dotProduct = __half2float(*(__half*)typed_dotProduct);
    LOG(INFO, global_state->rank)<<"typed dot product is: "<<*(__half*)typed_dotProduct<<" dot product is "<<dotProduct;
    
    anormsq = __half2float(*(__half*)typed_anormsq);
    LOG(INFO, global_state->rank)<<"typed anormsq is: "<<*(__half*)typed_anormsq<<" anormsq is "<<anormsq;

    bnormsq = __half2float(*(__half*)typed_bnormsq);
    LOG(INFO, global_state->rank)<<"typed_bnormsq is: "<<*(__half*)typed_bnormsq<<" bnormsq is "<<bnormsq;
  }
  else {
    dotProduct = (double)*(float *)typed_dotProduct;
    LOG(INFO, global_state->rank)<<"typed dot product is: "<<*(float *)typed_dotProduct<<" dot product is "<<dotProduct;
    
    anormsq = (double)*(float *)typed_anormsq;
    LOG(INFO, global_state->rank)<<"typed anormsq is: "<<*(float *)typed_anormsq<<" anormsq is "<<anormsq;

    bnormsq = (double)*(float *)typed_bnormsq;
    LOG(INFO, global_state->rank)<<"typed_bnormsq is: "<<*(float *)typed_bnormsq<<" bnormsq is "<<bnormsq;
  }
  delete typed_dotProduct;
  delete typed_anormsq;
  delete typed_bnormsq;
}


template<typename T>
void MsCudaAllreduceOp::ScaleAddImpl(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state, int layerid) {

  LOG(INFO, global_state->rank)<<" acoeff is "<<acoeff;

  cudaDataType_t cuda_type = CublasContext::GetCublasDataType(a);
  cudaDataType_t execution_type = cuda_type == CUDA_R_16F ? CUDA_R_32F : cuda_type;
  cudaDataType_t alpha_type = cuda_type == CUDA_R_64F ? CUDA_R_64F : CUDA_R_32F;
  float acoeff_float = (float)acoeff;
  auto scaleStatus = cublasScalEx(cublas_Handle, n, &acoeff_float, alpha_type, (void *)a, cuda_type, 1, execution_type);
  CublasContext::ErrorCheck("cublasScalEx", scaleStatus);
  
  float bcoeff_float = (float)bcoeff;
  auto axpyStatus = cublasAxpyEx(cublas_Handle, n, &bcoeff_float, alpha_type, (void *)b, cuda_type, 1, (void *)a, cuda_type, 1, execution_type);
  CublasContext::ErrorCheck("cublasAxpyEx", axpyStatus);
  cudaStream_t stream;
  cublasGetStream(cublas_Handle, &stream);
  auto cuda_sync_result = cudaStreamSynchronize(stream);
  CUDAContext::ErrorCheck("cudaStreamSynchronize", cuda_sync_result);
}
*/

bool MsCudaAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}
}
}
