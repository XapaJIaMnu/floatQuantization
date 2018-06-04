#include <vector>
#include <stdint.h>
#ifdef CUDA_FOUND
#include <cuda_fp16.h>
#endif

typedef union {
  // float8 v;
  struct {
    // type determines alignment!
    uint8_t m:4;
    uint8_t e:3;
    uint8_t s:1;
   } bits;
} float8_s;

typedef union {
  // float16 v;
  struct {
    // type determines alignment!
    uint16_t m:10;
    uint16_t e:5;
    uint16_t s:1;
   } bits;
} float16_s;

typedef union {
  float v;
  struct {
    uint32_t m:23;
    uint32_t e:8;
    uint32_t s:1;
   } bits;
} float32_s; 

#ifdef CUDA_FOUND
__device__ void float32to16_gpu(float x, float16_s * f16) {
  float32_s f32={x}; // c99

  // to 16
  f16->bits.s=f32.bits.s;
  f16->bits.e=max(-15, min(16,(int)(f32.bits.e-127))) +15;
  f16->bits.m=f32.bits.m >> 13;
}

__device__ float16_s float32to16_gpu(float x) {
  float32_s f32={x}; // c99
  float16_s f16;

  // to 16
  f16.bits.s=f32.bits.s;
  f16.bits.e=max(-15,min(16,(int)(f32.bits.e-127))) +15;
  f16.bits.m=f32.bits.m >> 13;
  return f16;
}
#endif

float16_s float32to16(float x) {
  float32_s f32={x}; // c99
  float16_s f16;

  // to 16
  f16.bits.s=f32.bits.s;
  f16.bits.e=std::max(-15,std::min(16,(int)(f32.bits.e-127))) +15;
  f16.bits.m=f32.bits.m >> 13;
  return f16;
}

void float32to16(float x, float16_s * f16) {
  float32_s f32={x}; // c99

  // to 16
  f16->bits.s=f32.bits.s;
  f16->bits.e=std::max(-15,std::min(16,(int)(f32.bits.e-127))) +15;
  f16->bits.m=f32.bits.m >> 13;
}
 
#ifdef CUDA_FOUND
__device__ __host__ float float16to32(float16_s f16) {
#else
float float16to32(float16_s f16) {
#endif
  // back to 32
  float32_s f32;
  f32.bits.s=f16.bits.s;
  f32.bits.e=(f16.bits.e-15)+127; // safe in this direction
  f32.bits.m=((uint32_t)f16.bits.m) << 13;
 
  return f32.v;
}
 

#ifdef CUDA_FOUND
__device__ void float32to8_gpu(float x, float8_s * f8) {
  float32_s f32={x}; // c99

  // to 8
  f8->bits.s=f32.bits.s;
  f8->bits.e=max(-3,min(4,(int)(f32.bits.e-127))) +3;
  f8->bits.m=f32.bits.m >> 19;
}

__device__ float8_s float32to8_gpu(float x) {
  float32_s f32={x}; // c99
  float8_s f8;

  // to 8
  f8.bits.s=f32.bits.s;
  f8.bits.e=max(-3,min(4,(int)(f32.bits.e-127))) +3;
  f8.bits.m=f32.bits.m >> 19;
  return f8;
}
#endif

void float32to8(float x, float8_s * f8) {
  float32_s f32={x}; // c99

  // to 8
  f8->bits.s=f32.bits.s;
  f8->bits.e=std::max(-3,std::min(4,(int)(f32.bits.e-127))) +3;
  f8->bits.m=f32.bits.m >> 19;
}

float8_s float32to8(float x) {
  float32_s f32={x}; // c99
  float8_s f8;

  // to 8
  f8.bits.s=f32.bits.s;
  f8.bits.e=std::max(-3,std::min(4,(int)(f32.bits.e-127))) +3;
  f8.bits.m=f32.bits.m >> 19;
  return f8;
}

#ifdef CUDA_FOUND 
__device__ __host__ float float8to32(float8_s f8) {
#else
float float8to32(float8_s f8) {
#endif
  // back to 32
  float32_s f32;
  f32.bits.s=f8.bits.s;
  f32.bits.e=(f8.bits.e-3)+127; // safe in this direction
  f32.bits.m=((uint32_t)f8.bits.m) << 19;
 
  return f32.v;
}

/*Assumes both vectors are allocated. */
void float32to8Vec(std::vector<float>& input, std::vector<float8_s>& output) {
  for (size_t i = 0; i < input.size(); i++) {
    float32to8(input[i], &output[i]);
  }
}

void float8to32Vec(std::vector<float8_s>& input, std::vector<float>& output) {
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = float8to32(input[i]);
  }
}

/*Assumes both vectors are allocated. */
void float32to16Vec(std::vector<float>& input, std::vector<float16_s>& output) {
  for (size_t i = 0; i < input.size(); i++) {
    float32to16(input[i], &output[i]);
  }
}

void float16to32Vec(std::vector<float16_s>& input, std::vector<float>& output) {
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = float16to32(input[i]);
  }
}

/*Assumes both vectors are allocated. */
void float32to8Arr(float * input, float8_s * output, size_t count) {
  for (size_t i = 0; i < count; i++) {
    float32to8(input[i], &output[i]);
  }
}

void float8to32Arr(float8_s * input, float * output, size_t count) {
  for (size_t i = 0; i < count; i++) {
    output[i] = float8to32(input[i]);
  }
}

/*Assumes both vectors are allocated. */
void float32to16Arr(float * input, float16_s * output, size_t count) {
  for (size_t i = 0; i < count; i++) {
    float32to16(input[i], &output[i]);
  }
}

void float16to32Arr(float16_s * input, float * output, size_t count) {
  for (size_t i = 0; i < count; i++) {
    output[i] = float16to32(input[i]);
  }
}

//CUDA kernels
#ifdef CUDA_FOUND
__global__ void float32to8Kern(float * input, float8_s * output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float32to8_gpu(input[idx], &output[idx]);
}

__global__ void float8to32Kern(float8_s * input, float * output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    output[idx] = float8to32(input[idx]);
}

__global__ void float32to16Kern(float * input, float16_s * output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float32to16_gpu(input[idx], &output[idx]);
}

__global__ void float16to32Kern(float16_s * input, float * output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    output[idx] = float16to32(input[idx]);
}

__global__ void float32to16Kern(float * input, __half * output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    output[idx] = __float2half(input[idx]);
}

__global__ void float16to32Kern(__half * input, float * output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    output[idx] = __half2float(input[idx]);
}
#endif
