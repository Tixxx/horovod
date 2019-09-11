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

#ifndef HOROVOD_P2P_OPERATIONS_H
#define HOROVOD_P2P_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"
#include "collective_operations.h"


namespace horovod {
namespace common {

class PointToPointOp : public AllreduceOp {
public:
  PointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state);

  virtual ~PointToPointOp() = default;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;

  template<typename T>
  void MpiP2pAllreduce(T *grad_buffer, T *recv_buffer, int64_t buffer_length, MPI_Comm communicator, int message_tag) {
    int true_rank;
    int redn_rank;
    int size;
    MPI_Comm_rank(communicator, &true_rank);
    MPI_Comm_size(communicator, &size);

    static bool opt_permute_roots_on_allreduce = false;
    int root_node_rotation = opt_permute_roots_on_allreduce ? (message_tag % size) : 0;
    redn_rank = (true_rank ^ root_node_rotation);
    
    int count = buffer_length/sizeof(T);
    assert(!(grad_buffer <= recv_buffer && recv_buffer < grad_buffer + count) || (recv_buffer <= grad_buffer && grad_buffer < recv_buffer + count));

    // Do a tree reduction
    // The reduction ranks used are a permutation of true ranks (permuted based on message_tag)
    // This spreads the load of tree reduction across different true ranks

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
    PointToPointRecv(recv_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);

    // do reduction
    for(int i = 0; i < count; i++) {
    grad_buffer[i] += recv_buffer[i];
    }
    }
    else {
    // send grad_buffer to neighbor
    PointToPointSend(grad_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);
    }
    }

    // Do a inverse tree to do a broadcast
    // cannot use MPI Broadcast as there can be concurrent Allreduces happening in parallel

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
      PointToPointSend(grad_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);
      }
      else {
      // recv grad_buffer from neighbor
      PointToPointRecv(grad_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);
      }
    }
  }

  template<class T>
  void PointToPointSend(T* input_data_buffer,
                        int64_t buffer_length,
                        int dest_rank,
                        int tag,
                        MPI_Comm communicator) {
    int status;                       
    if (!global_state_->msg_chunk_enabled) {
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" begin p2p send for tag: "<<tag;
        status = MPI_Send(input_data_buffer,
                          (int)buffer_length,
                          MPI_CHAR,
                          dest_rank,
                          tag,
                          communicator);
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" end p2p send for tag: "<<tag;

    }
    else {
          const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
          for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
            status = MPI_Send((uint8_t *)input_data_buffer + buf_index,
                              std::min((int)buffer_length - buf_index, chunk_size) * sizeof(T),
                              MPI_CHAR,
                              dest_rank,
                              tag,
                              communicator);
            status &= status;
          }
    }

    if (status != MPI_SUCCESS) {
      throw std::logic_error("MPI_Send failed, see MPI output for details.");
    }
  }

  template<class T>
  void PointToPointRecv(T* output_data_buffer,
                        int64_t buffer_length,
                        int src_rank,
                        int tag,
                        MPI_Comm communicator) {
    int status;
    if (!global_state_->msg_chunk_enabled) {
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" begin p2p recv for tag: "<<tag;
        status = MPI_Recv(output_data_buffer,
                          (int)buffer_length,
                          MPI_CHAR,
                          src_rank,
                          tag,
                          communicator,
                          MPI_STATUS_IGNORE);
        LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" end p2p recv for tag: "<<tag;
    }
    else {
          const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
          for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
            status = MPI_Recv((uint8_t *)output_data_buffer + buf_index,
                              std::min((int)buffer_length - buf_index, chunk_size) * sizeof(T),
                              MPI_CHAR,
                              src_rank,
                              tag,
                              communicator,
                              MPI_STATUS_IGNORE);
            status &= status;
          }
    }

    if (status != MPI_SUCCESS) {
      throw std::logic_error("MPI_Recv failed, see MPI output for details.");
    }
  }

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_P2P_OPERATIONS_H
