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
#ifndef HOROVOD_MSALLREDUCE_CUDA_OPERATIONS_H
#define HOROVOD_MSALLREDUCE_CUDA_OPERATIONS_H

#include "msallreduce_operations.h"
#include "cuda_operations.h"
#include "cublas_v2.h"

namespace horovod {
namespace common {
class CublasContext {
  public:
  std::string static GetCublasErrorString (cublasStatus_t cublas_result);

  void static ErrorCheck(std::string op_name, cublasStatus_t cublas_result);
};

struct CublasThreadState {

    public:

    cublasHandle_t cublasHandle;

    CublasThreadState() {
        auto status = cublasCreate(&cublasHandle);
        CublasContext::ErrorCheck("cublasCreate", status);
    }

    ~CublasThreadState() {
        auto status = cublasDestroy(cublasHandle);
        CublasContext::ErrorCheck("cublasDestroy", status);
    }
};

class MsCudaAllreduceOp : public MsAllreduceOp {
  public:
  MsCudaAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context,
                HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  protected:
  struct CUDAContext* cuda_context_;

  CublasThreadState static getCublasThreadState();

  void memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len) override;

  template<typename T>
  void static DotProductImpl(const T* __restrict__  a, const T* __restrict__ b, int n, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state);
  
  template<typename T>
  void static ScaleAddImpl(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state);
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_MSALLREDUCE_CUDA_OPERATIONS_H