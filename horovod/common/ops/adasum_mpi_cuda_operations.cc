// Copyright 2019 Microsoft. All Rights Reserved.
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

#include "adasum_mpi_cuda_operations.h"

namespace horovod {
namespace common {
AdasumMPICudaAllreduceOp::AdasumMPICudaAllreduceOp(MPIContext* mpi_context, CUDAContext* context,
                                           HorovodGlobalState* global_state)
    : AdasumMPI(mpi_context, global_state), CUDAAllreduce(context, global_state) {}

AdasumMPICudaAllreduceOp::~AdasumMPICudaAllreduceOp() {
  if (device_normsq_a != nullptr) {
    cuda_context_->ErrorCheck("cudaFree",
      cudaFree(device_normsq_a));
    cuda_context_->ErrorCheck("cudaFree",
      cudaFree(device_normsq_b));
    cuda_context_->ErrorCheck("cudaFree",
      cudaFree(device_dot));
    device_normsq_a = nullptr;
    device_normsq_b = nullptr;
    device_dot = nullptr;
  }
}

bool AdasumMPICudaAllreduceOp::Enabled(const ParameterManager& param_manager,
                                   const std::vector<TensorTableEntry>& entries,
                                   const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

uint8_t* AdasumMPICudaAllreduceOp::CheckBufferAndReallocate(uint8_t** buffer,
                                          uint64_t buffer_length,
                                          uint64_t& current_length) {
  if (buffer_length <= current_length) {
    return *buffer;
  }
  if (*buffer != nullptr) {
    cudaFree(*buffer);
  }
  cuda_context_->ErrorCheck("cudaMalloc",
    cudaMalloc((void**)buffer, buffer_length));
  current_length = buffer_length;
  return *buffer;
}

void AdasumMPICudaAllreduceOp::FreeBuffer(uint8_t** buffer) {
  cuda_context_->ErrorCheck("cudaFree",
    cudaFree(*buffer));
  *buffer = nullptr;
}

Status AdasumMPICudaAllreduceOp::Execute(std::vector<TensorTableEntry>& entries,
                                     const Response& response) {
  if (entries.empty()) {
    return Status::OK();
  }

  // Lazily initialize reduction communicators for VHDD algorithm when Adasum reduction is actually called.
  if (!reduction_comms_initialized) {
    InitializeVHDDReductionComms();
  }

  auto& first_entry = entries[0];

  cuda_op_context_.InitCUDA(entries);
  InitDeviceVariables();

  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
    if (first_entry.tensor->data() != first_entry.output->data()) {
      auto cuda_result = cudaMemcpyAsync(buffer_data, (void*)first_entry.tensor->data(),
                                         (size_t) first_entry.tensor->size(), cudaMemcpyDeviceToDevice,
                                         cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
      cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
      cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
      cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
    }
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ADASUM_ALLREDUCE);
  std::vector<int> tensor_counts;
  for (auto& e : entries) {
    tensor_counts.push_back(e.tensor->shape().num_elements());
  }

  auto recv_buffer = GetRecvBuffer(buffer_len);
  DispatchFusedAllreduce(entries, buffer_data, recv_buffer, tensor_counts,
                         1, // start_level
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL),
                         0, // tag
                         reduction_comms_, first_entry.tensor->dtype(),
                         global_state_);
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

void AdasumMPICudaAllreduceOp::DispatchComputeDotAndNormSqrds(const void* __restrict__ a,
                                                              const void* __restrict__ b,
                                                              DataType horovod_datatype,
                                                              int count, double& dotProduct,
                                                              double& anormsq, double& bnormsq,
                                                              int layerid) {
  if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
    CudaDotProductImpl(count, (uint16_t*)a, (uint16_t*)b, device_normsq_a, device_normsq_b, device_dot, anormsq, bnormsq, dotProduct);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    CudaDotProductImpl(count, (float*)a, (float*)b, device_normsq_a, device_normsq_b, device_dot, anormsq, bnormsq, dotProduct);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
    CudaDotProductImpl(count, (double*)a, (double*)b, device_normsq_a, device_normsq_b, device_dot, anormsq, bnormsq, dotProduct);
  }
  else {
    throw std::logic_error("Unsupported data type.");
  }
}

void AdasumMPICudaAllreduceOp::DispatchScaledAdd(DataType horovod_datatype, int count,
                                                 double acoeff, void* __restrict__ a,
                                                 double bcoeff, void* __restrict__ b,
                                                 int layerid) {
  if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
    CudaScaleAddImpl(count, (uint16_t*)a, (uint16_t*)b, acoeff, bcoeff);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    CudaScaleAddImpl(count, (float*)a, (float*)b, acoeff, bcoeff);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
    CudaScaleAddImpl(count, (double*)a, (double*)b, acoeff, bcoeff);
  }
  else {
    throw std::logic_error("Unsupported data type.");
  }
}

void AdasumMPICudaAllreduceOp::InitDeviceVariables() {
  if (device_normsq_a == nullptr) {
    cuda_context_->ErrorCheck("cudaMalloc",
      cudaMalloc(&device_normsq_a, sizeof(double)));
    cuda_context_->ErrorCheck("cudaMalloc",
      cudaMalloc(&device_normsq_b, sizeof(double)));
    cuda_context_->ErrorCheck("cudaMalloc",
      cudaMalloc(&device_dot, sizeof(double)));
  }
}
} // namespace common
} // namespace horovod
