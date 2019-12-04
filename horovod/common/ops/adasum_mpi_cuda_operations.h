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

#ifndef HOROVOD_ADASUM_MPI_OPERATIONS_H
#define HOROVOD_ADASUM_MPI_OPERATIONS_H

#include "mpi.h"
#include <iostream>

#include "adasum/adasum_mpi.h"
#include "cuda/adasum_cuda_kernels.h"
#include "cuda_operations.h"

namespace horovod {
namespace common {

class AdasumMPICudaAllreduceOp : public AdasumMPI, public CUDAAllreduce {
public:
  AdasumMPICudaAllreduceOp(MPIContext* mpi_context, CUDAContext* context,
                       HorovodGlobalState* global_state);
  ~AdasumMPICudaAllreduceOp();

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  uint8_t* CheckBufferAndReallocate(uint8_t** buffer,
                                    uint64_t buffer_length,
                                    uint64_t& current_length) override;

  void InitDeviceVariables();

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  void DispatchComputeDotAndNormSqrds(const void* __restrict__ a,
                                      const void* __restrict__ b,
                                      DataType horovod_datatype,
                                      int count, double& dotProduct,
                                      double& anormsq, double& bnormsq,
                                      int layerid) override;

  void DispatchScaledAdd(DataType horovod_datatype, int count,
                         double acoeff, void* __restrict__ a,
                         double bcoeff, void* __restrict__ b,
                         int layerid) override; };
private:
  double* device_normsq_a;
  double* device_normsq_b;
  double* device_dot;
} // namespace common
} // namespace horovod

#endif // HOROVOD_ADASUM_MPI_OPERATIONS_H
