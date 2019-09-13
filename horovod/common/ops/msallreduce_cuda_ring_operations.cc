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

#include "msallreduce_cuda_ring_operations.h"
#include "msallreduce_cuda_kernels.h"

namespace horovod {
namespace common {

using namespace msallreduce;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	} 
}

MsCudaRingAllreduceOp::MsCudaRingAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : MsAllreduceOp(mpi_context, global_state), mpi_context_(mpi_context), cuda_context_(cuda_context) {
    }

void MsCudaRingAllreduceOp::InitCUDA(const TensorTableEntry& entry, int layerid) {
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));

  LOG(INFO, global_state_->rank)<<"Checking for existing stream for layer "<<layerid<<" "<<std::this_thread::get_id();
  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[global_state_->current_nccl_stream][layerid];
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
    if (device_stream == nullptr) {
      LOG(INFO, global_state_->rank)<<"device Stream is null, creating new device stream "<<std::this_thread::get_id();
      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&device_stream, cudaStreamNonBlocking, greatest_priority));

    }
  }
}

Status MsCudaRingAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  if(entries.size() < 1) {
      return Status::OK();
  }
  //TODO how do we report statuses?
  std::map<int, Status> return_statuses;
  int num_reductions = entries.size();
	AllRings all_rings(global_state_->local_rank, global_state_->local_size);
  std::deque<FusionBufferManager> used_buffer_managers;
  std::deque<void*> recv_buffers;
  LOG(INFO, global_state_->rank)<<"Ready to process "<<num_reductions<<" tensors in gpu";
  size_t unroll_size = 8;
  size_t layerid = 0;
  size_t elements_left = num_reductions;
  global_state_->finished_parallel_reductions = 0;
  while(elements_left > 0){
    size_t increment_count = std::min(unroll_size, elements_left);
    size_t start_index = layerid;
    //enqueue messages
    for (; layerid < layerid + increment_count; ++layerid) {
      auto& entry = entries.at(layerid);
      void* buffer_data;
      int buffer_len;
      void* recv_buffer;

      buffer_data = (void*) entry.tensor->data();

      buffer_len = entry.output->size();

      if(entry.tensor->data() == entry.output->data()) {
          LOG(INFO, global_state_->rank)<<"Output and input pointing to same data. Creating temp buffer "<<std::this_thread::get_id();

          // Get the temp buffer to be used for the Op
          FusionBufferManager buffer_manager;
          if (!buffer_managers_.empty()) {
            buffer_manager = buffer_managers_.front();
            buffer_managers_.pop_front();
          }
          used_buffer_managers.push_back(buffer_manager);

          // TODO: Maybe add before and after callbacks to timeline?
          Status status = buffer_manager.InitializeBuffer(
              buffer_len,
              entry.device, entry.context,
              global_state_->current_nccl_stream,
              []() {},
              []() {},
              [](int64_t& size, int64_t& threshold) {return size >= threshold;});

          if (!status.ok()) {
              throw std::logic_error("MsAllreduceOp::Execute_helper: Initialize buffer failed.");
          }
          auto& buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
          recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
      }
      else {
          recv_buffer = (void*) entry.output->data();
      }
      recv_buffers.push_back(recv_buffer);
      LOG(INFO, global_state_->rank)<<"Begin to process gpu tensor with size "<<entry.tensor->size()<<" into output buffer with size "<<entry.output->size()<<" "<<std::this_thread::get_id();
    
      // This will create a stream per inner layer.
      InitCUDA(entry, layerid);
      LOG(INFO, global_state_->rank)<<"Begin processing gpu tensor in layer "<<layerid<<" "<<std::this_thread::get_id();
      all_rings.InitMessageInRing(new ReduceMessage(mpi_context_),
                        buffer_data,
                        recv_buffer,
                        buffer_len,
                        entry.tensor->dtype(),
                        global_state_->local_comm,
                        layerid,
                        global_state_->local_rank);
    }

    // wait for messages to finish
    all_rings.WaitAllMessages();
    LOG(INFO, global_state_->rank)<<"All messages are done "<<" "<<std::this_thread::get_id();
    // Return used buffer managers to the queue
    buffer_managers_.insert(buffer_managers_.end(), used_buffer_managers.begin(), used_buffer_managers.end());
  
    int local_rank = 0;
    // cross-node vhdd reduction
    MPI_Comm_rank(global_state_->local_comm, &local_rank);
    if (local_rank == 0 && global_state_->rank_log_size != 0) {
      boost::asio::post(*global_state_->background_thread_pool,
      [this,&entries, &start_index, &increment_count]
      {
        LOG(INFO, global_state_->rank)<<"Begin vhdd"<<" "<<std::this_thread::get_id();
        std::vector<std::unique_ptr<char[]>> allreduce_buffers;

        // start device to host copies
        for (size_t index = start_index; index < start_index + increment_count; ++index) {
          auto& entry = entries.at(index);
          int buffer_len = entry.output->size();
          allreduce_buffers.emplace_back(new char[buffer_len]);
          char* buffer_data = allreduce_buffers.at(index).get();
          
          auto cuda_result = cudaMemcpyAsync(
            buffer_data, (void*) entry.tensor->data(),
            buffer_len, 
            cudaMemcpyDeviceToHost,
            cuda_context_->streams[global_state_->current_nccl_stream][index]);
          cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
        }
        LOG(INFO, global_state_->rank)<<"Finished dispatching cudamemcopyasync"<<" "<<std::this_thread::get_id();

        for (size_t index = start_index; index < start_index + increment_count; ++index) {
          auto& entry = entries.at(index);
          int buffer_len = entry.output->size();
          char* buffer_data = allreduce_buffers.at(index).get();
          std::unique_ptr<char[]> recv_buffer(new char[buffer_len]);

          // wait for this layer to finish copying to host
          auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][index]);
          cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

          MPI_Comm* node_comm = &global_state_->reduction_comms[global_state_->rank_log_size-1];
          switch (entry.output->dtype()) {
              case HOROVOD_FLOAT16:
                SyncAllreduce(
                  (uint16_t*) buffer_data,
                  (uint16_t*) recv_buffer.get(),
                  buffer_len / sizeof(uint16_t),
                  *node_comm,
                  global_state_->reduction_comms,
                  index,
                  entry,
                  ComputeDotAndNormSqrdsfp16,
                  ScaledAddfp16);
              break;
              case HOROVOD_FLOAT32:
                SyncAllreduce(
                  (float*) buffer_data,
                  (float*) recv_buffer.get(),
                  buffer_len / sizeof(float),
                  *node_comm,
                  global_state_->reduction_comms,
                  index,
                  entry,
                  ComputeDotAndNormSqrds<float>,
                  ScaledAdd<float>);
              break;
              case HOROVOD_FLOAT64:
                SyncAllreduce(
                  (double*) buffer_data,
                  (double*) recv_buffer.get(),
                  buffer_len / sizeof(double),
                  *node_comm,
                  global_state_->reduction_comms,
                  index,
                  entry,
                  ComputeDotAndNormSqrds<double>,
                  ScaledAdd<double>);
              break;
              default:
                throw std::logic_error("MsAllreduceOp::Execute: Unsupported data type.");
          }
          LOG(INFO, global_state_->rank)<<"Finished syncAllreduce"<<" "<<std::this_thread::get_id();

          // start the copy back to device
          cuda_result = cudaMemcpyAsync(
            (void*) entry.tensor->data(), buffer_data,
            buffer_len, 
            cudaMemcpyHostToDevice,
            cuda_context_->streams[global_state_->current_nccl_stream][index]);
          cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
        }
        LOG(INFO, global_state_->rank)<<"Cudamemcpy to device dispatched"<<" "<<std::this_thread::get_id();

        // wait for all copies to device to finish
        for (size_t index = start_index; index < start_index + increment_count; ++index) {
          auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][index]);
          cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
        }
        LOG(INFO, global_state_->rank)<<"Tensors copied back to device"<<" "<<std::this_thread::get_id();
        global_state_->finished_parallel_reductions++;
      });
    }
    // for ranks that are not doing vhdd, increment finished_parallel_reductions right away
    else {
      global_state_->finished_parallel_reductions++;
    }
    elements_left -= increment_count;
  }

  // wait for all vhdd to finish
  while (global_state_->finished_parallel_reductions.load() < num_reductions) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(25));
  }
  MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  //ring broadcast
  for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
    auto& entry = entries.at(layerid);
    void* buffer_data;
    int buffer_len;

    buffer_data = (void*) entry.tensor->data();

    buffer_len = entry.output->size();

    LOG(INFO, global_state_->rank)<<"Begin to process gpu tensor with size "<<entry.tensor->size()<<" into output buffer with size "<<entry.output->size()<<" "<<std::this_thread::get_id();
  
    // This will create a stream per layer.
    //InitCUDA(entry, layerid);
    LOG(INFO, global_state_->rank)<<"Begin processing gpu tensor in layer "<<layerid<<" "<<std::this_thread::get_id();
    all_rings.InitMessageInRing(new BroadcastMessage(mpi_context_),
                      buffer_data,
                      nullptr,
                      buffer_len,
                      entry.output->dtype(),
                      global_state_->local_comm,
                      layerid,
                      global_state_->local_rank);
  }
  all_rings.WaitAllMessages();
  for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
    auto& entry = entries.at(layerid);
    if(entry.tensor->data() != entry.output->data()) {
      memcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size(), layerid);
    }
  }

  return Status::OK();
}

void MsCudaRingAllreduceOp::memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) {
    assert(dest != nullptr);
    assert(src != nullptr);
    LOG(INFO, global_state_->rank)<<"memcpyUtil GPU. "<<std::this_thread::get_id();
    auto cuda_result = cudaMemcpyAsync(dest, src,
                                    buffer_len, 
                                    cudaMemcpyDeviceToDevice,
                                    cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    auto cuda_sync_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_sync_result);
}

bool MsCudaRingAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

namespace msallreduce{

void Ring::InitRing(int tmp[], bool _isFat, int rank, int size) {
  load = 0;
  isFat = _isFat;
  for (int i = 0; i < 8; i++)
    loop[i] = tmp[i];

  for (int j = 0; j < size; j++) { // go through allranks
    if (rank == loop[j]) {
      prevGPU = loop[(j-1+size) % size];
      nextGPU = loop[(j+1+size) % size];
    }
  }
}

int Ring::GetAfterLoad(int message_len) {
  if (!isFat)
    return 2*(load+message_len);
  else
    return (load+message_len);
}

void Ring::AddLoad(int message_len) {
  load += message_len;
}

void Ring::ReduceLoad(int message_len) {
  load -= message_len;
}

Message::Message(MPIContext* mpi_context)
  : mpi_context(mpi_context) {
}

void Message::InitMessage(Ring* _ring, int _rank, int _ring_starter_rank, int _count, void* _grad_buf, void* _recv_buf, DataType _datatype, MPI_Comm _comm, int _tag) {
  comm = _comm;
  count = _count;
  tag = _tag;
  ring = _ring;
  rank = _rank;
  ring_starter_rank = _ring_starter_rank;
  leg = 0;
  grad_buf = _grad_buf;
  recv_buf = _recv_buf;
  datatype = _datatype;
  Start();
}

AllreduceMessage::AllreduceMessage(MPIContext* mpi_context)
  : Message(mpi_context) {
}

void AllreduceMessage::Start() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);
  if (rank == ring_starter_rank) {
    MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
  } else {
    MPI_Irecv(recv_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
  }
}

bool AllreduceMessage::Test() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);

  int flag;
  if (leg == 4)
    return true;
  MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  if (flag == 1) {
    leg++;
    if (leg == 4) {
      ring->ReduceLoad(count);
      return true;
    }
    if (leg == 1) {
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else {
        // call the cuda kernel
        switch(datatype) {
          case HOROVOD_FLOAT16:
            MsCudaPairwiseReduce(count, (uint16_t*)grad_buf, (uint16_t*)recv_buf);
            break;
          case HOROVOD_FLOAT32:
            MsCudaPairwiseReduce(count, (float*)grad_buf, (float*)recv_buf);
            break;
          case HOROVOD_FLOAT64:
            MsCudaPairwiseReduce(count, (double*)grad_buf, (double*)recv_buf);
            break;
          default:
            throw std::logic_error("Message::Test: Unsupported data type.");
        }
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    } else if (leg == 2) {
      if (rank == ring_starter_rank) {
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      } else {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      }
    } else if (leg == 3) {
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else {
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    }
  }

  return false;
}

ReduceMessage::ReduceMessage(MPIContext* mpi_context)
  : Message(mpi_context) {
}

void ReduceMessage::Start() {

  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);
  if (rank == ring_starter_rank) {
    MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
  } else {
    MPI_Irecv(recv_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
  }
}

bool ReduceMessage::Test() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);

  int flag;
  if (leg == 2)
    return true;
  MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  if (flag == 1) {
    leg++;
    if (leg == 2) {
      ring->ReduceLoad(count);
      return true;
    }
    if (leg == 1) {
      if (rank == ring_starter_rank) {
        MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
      } else {
        // call the cuda kernel
        switch(datatype) {
          case HOROVOD_FLOAT16:
            MsCudaPairwiseReduce(count, (uint16_t*)grad_buf, (uint16_t*)recv_buf);
            break;
          case HOROVOD_FLOAT32:
            MsCudaPairwiseReduce(count, (float*)grad_buf, (float*)recv_buf);
            break;
          case HOROVOD_FLOAT64:
            MsCudaPairwiseReduce(count, (double*)grad_buf, (double*)recv_buf);
            break;
          default:
            throw std::logic_error("Message::Test: Unsupported data type.");
        }
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    }
  }
  return false;
}

BroadcastMessage::BroadcastMessage(MPIContext* mpi_context)
  : Message(mpi_context) {
}

void BroadcastMessage::Start() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);
  if (rank == ring_starter_rank) {
    MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
    leg = 1;
  } else {
    MPI_Irecv(grad_buf, count, mpi_datatype, ring->prevGPU, tag, comm, &req);
    if (ring->nextGPU == ring_starter_rank)
      leg = 1;
  }
}

bool BroadcastMessage::Test() {
  auto mpi_datatype = mpi_context->GetMPIDataType(datatype);

  int flag;
  if (leg == 2)
    return true;
  MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  if (flag == 1) {
    leg++;
    if (leg == 2) {
      ring->ReduceLoad(count);
      return true;
    }
    if (leg == 1) {
      if (rank != ring_starter_rank) {
        MPI_Isend(grad_buf, count, mpi_datatype, ring->nextGPU, tag, comm, &req);
      }
    }
  }
  return false;
}

AllRings::~AllRings() {
  for (int i = 0; i < messages.size(); i++)
    delete messages[i];
  delete[] rings;
}

AllRings::AllRings(int rank, int size) {
  rings = new Ring[num_rings];
  {
    // fat ring 1
    int tmp[8] = {0, 3, 2, 1, 5, 6, 7, 4};
    rings[0].InitRing(tmp, true, rank, size);
  }
  {
    // fat ring 2
    int tmp[8] = {0, 4, 7, 6, 5, 1, 2, 3};
    rings[1].InitRing(tmp, true, rank, size);
  }
  {
    // skinny ring 1
    int tmp[8] = {0, 2, 6, 4, 5, 7, 3, 1};
    rings[2].InitRing(tmp, false, rank, size);
  }
  {
    // skinny ring 2
    int tmp[8] = {0, 1, 3, 7, 5, 4, 6, 2};
    rings[3].InitRing(tmp, false, rank, size);
  }
};

Ring* AllRings::PickRing(int count) {
  int min_load = (1<<30); // INF
  Ring* ret_ring = NULL;
  for (int i = 0; i < num_rings; i++) {
    Ring* ring = &rings[i];
    int cur_ring_after_load = ring->GetAfterLoad(count);
    if (cur_ring_after_load < min_load) {
      ret_ring = ring;
      min_load = cur_ring_after_load;
    }
  }
  ret_ring->AddLoad(count);
  assert(ret_ring != NULL);
  return ret_ring;
}

void AllRings::InitMessageInRing(Message* message, void* grad_buf, void* recv_buf, int size, DataType datatype, MPI_Comm comm, int grad_tag, int rank) {
  int count = -1;
  switch(datatype) {
    case HOROVOD_FLOAT16:
      count = size / sizeof(uint16_t);
      break;
    case HOROVOD_FLOAT32:
      count = size / sizeof(float);
      break;
    case HOROVOD_FLOAT64:
      count = size / sizeof(double);
      break;
    default:
      throw std::logic_error("AllRings::InitMessageInRing: Unsupported data type.");
  }
  messages.push_back(message);
	Ring*	ring = PickRing(count);
  message->InitMessage(ring, rank, grad_tag % 8, count, grad_buf, recv_buf, datatype, comm, grad_tag);
}

void AllRings::WaitAllMessages() {
  
  bool all_done = false;
  while (!all_done) {
    all_done = true;
    for (auto& message : messages) {
      if (!message->Test())
        all_done = false;
    }
  }
  for (int i = 0; i < messages.size(); i++)
    delete messages[i];
  messages.clear();
}

}
}
}
