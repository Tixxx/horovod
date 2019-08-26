#include <stdint.h>

void CudaDotProductImpl(int count, const double* device_a, const double* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot);

void CudaDotProductImpl(int count, const float* device_a, const float* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot);

void CudaDotProductImpl(int count, const uint16_t* device_a, const uint16_t* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot);

void CudaScaleAddImpl(int count, double* a_device, const double* b_device, double host_a_coeff, double host_b_coeff);

void CudaScaleAddImpl(int count, float* a_device, const float* b_device, double host_a_coeff, double host_b_coeff);

void CudaScaleAddImpl(int count, uint16_t* a_device, const uint16_t* b_device, double host_a_coeff, double host_b_coeff);