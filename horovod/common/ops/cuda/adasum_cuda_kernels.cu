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

#include <iostream>
#include <stdio.h>
#include <cuda_fp16.h>
#include <time.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 64

template<typename T, typename TACC>
__global__
void CudaDotProductKernel(int count, const T* a, const T* b, TACC* out_normsq_a, TACC* out_normsq_b, TACC* out_dot) {
	__shared__ TACC normsq_a[THREADS_PER_BLOCK];
	__shared__ TACC normsq_b[THREADS_PER_BLOCK];
	__shared__ TACC dot[THREADS_PER_BLOCK];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < count){
		normsq_a[threadIdx.x] = (TACC) a[index] * (TACC) a[index];
		normsq_b[threadIdx.x] = (TACC) b[index] * (TACC) b[index];
		dot[threadIdx.x]      = (TACC) a[index] * (TACC) b[index];
	}
	__syncthreads();
	if (0 == threadIdx.x) {
		TACC normsq_a_sum = 0;
		TACC normsq_b_sum = 0;
		TACC dot_sum = 0;
		for(int i = 0; i < THREADS_PER_BLOCK; i++){
			if (i + blockIdx.x * blockDim.x < count){
				normsq_a_sum += normsq_a[i];
				normsq_b_sum += normsq_b[i];
				dot_sum += dot[i];
			}
		}
		atomicAdd(out_normsq_a, normsq_a_sum);
		atomicAdd(out_normsq_b, normsq_b_sum);
		atomicAdd(out_dot, dot_sum);
	}
}

template<typename T, typename TACC>
__global__
void CudaScaleAddKernel(int count, T* a, const T* b, TACC a_coeff, TACC b_coeff) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (count > index){
		a[index] = (T) ((TACC) a[index] * a_coeff + (TACC) b[index] * b_coeff);
	}
}

template<typename T>
__global__
void ConvertToFloat(int count, T* a, float* b) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (count > index){
		b[index] = (float) a[index];
	}
}

void ErrorCheck(std::string op_name, cudaError_t cuda_result) {
	if (cuda_result != cudaSuccess) {
		throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
	}
}

void CudaDotProductImpl(int count, const double* device_a, const double* device_b, 
	double* device_normsq_a, double* device_normsq_b, double* device_dot, double& host_normsq_a, double& host_normsq_b, double& host_dot) {
	
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_normsq_a, &host_normsq_a, sizeof(double), cudaMemcpyHostToDevice));
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_normsq_b, &host_normsq_b, sizeof(double), cudaMemcpyHostToDevice));
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_dot, &host_dot, sizeof(double), cudaMemcpyHostToDevice));

	CudaDotProductKernel<<<(count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK>>>(count, device_a, device_b, device_normsq_a, device_normsq_b, device_dot);
	ErrorCheck("CudaDotProductKernel(double)", cudaGetLastError());
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_normsq_a, device_normsq_a, sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_normsq_b, device_normsq_b, sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_dot, device_dot, sizeof(double), cudaMemcpyDeviceToHost));

}

void CudaDotProductImpl(int count, const float* device_a, const float* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot, double& host_normsq_a, double& host_normsq_b, double& host_dot) {
	
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_normsq_a, &host_normsq_a, sizeof(double), cudaMemcpyHostToDevice));
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_normsq_b, &host_normsq_b, sizeof(double), cudaMemcpyHostToDevice));
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_dot, &host_dot, sizeof(double), cudaMemcpyHostToDevice));

	CudaDotProductKernel<<<(count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK>>>(count, device_a, device_b, device_normsq_a, device_normsq_b, device_dot);
	ErrorCheck("CudaDotProductKernel(float)", cudaGetLastError());
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_normsq_a, device_normsq_a, sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_normsq_b, device_normsq_b, sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_dot, device_dot, sizeof(double), cudaMemcpyDeviceToHost));

}

void CudaDotProductImpl(int count, const uint16_t* device_a, const uint16_t* device_b, 
	double* device_normsq_a, double* device_normsq_b, double* device_dot, double& host_normsq_a, double& host_normsq_b, double& host_dot) {
	
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_normsq_a, &host_normsq_a, sizeof(double), cudaMemcpyHostToDevice));
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_normsq_b, &host_normsq_b, sizeof(double), cudaMemcpyHostToDevice));
	ErrorCheck("cudaMemcpy", cudaMemcpy(device_dot, &host_dot, sizeof(double), cudaMemcpyHostToDevice));

	CudaDotProductKernel<<<(count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,
		THREADS_PER_BLOCK>>>(count, (__half*) device_a, (__half*) device_b, device_normsq_a, device_normsq_b, device_dot);
	ErrorCheck("CudaDotProductKernel(fp16)", cudaGetLastError());
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_normsq_a, device_normsq_a, sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_normsq_b, device_normsq_b, sizeof(double), cudaMemcpyDeviceToHost));
	ErrorCheck("cudaMemcpy", cudaMemcpy(&host_dot, device_dot, sizeof(double), cudaMemcpyDeviceToHost));

}

void CudaScaleAddImpl(int count, double* a_device, const double* b_device, double host_a_coeff, double host_b_coeff) {
	CudaScaleAddKernel<<<(count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(count, a_device, b_device,
		host_a_coeff, host_b_coeff);
}

void CudaScaleAddImpl(int count, float* a_device, const float* b_device, double host_a_coeff, double host_b_coeff) {
	CudaScaleAddKernel<<<(count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(count, a_device, b_device,
		host_a_coeff, host_b_coeff);
}

void CudaScaleAddImpl(int count, uint16_t* a_device, const uint16_t* b_device, double host_a_coeff, double host_b_coeff) {
	CudaScaleAddKernel<<<(count+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(count, (__half*)a_device, (__half*)b_device,
		host_a_coeff, host_b_coeff);
}
