#include <stdint.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#define CUDA_FOUND
#include "floatUnion.h"

#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if(code != cudaSuccess) {
    printf("Error: %s, %s, %s", cudaGetErrorString(code), file, line);
    std::abort();
  }
}

int main() {
    std::vector<float> base_vector{-4.6991462f, 8.8798271f, -0.38823462f, 0.13823462f, -1.16823462f};
    float * gpuMemInput;
    float8_s * gpuMemOutput;
    CUDA_CHECK(cudaMalloc(&gpuMemInput, 5*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpuMemOutput, 5*sizeof(float8_s)));
    CUDA_CHECK(cudaMemcpy(gpuMemInput, base_vector.data(), 5*sizeof(float), cudaMemcpyDefault));
    float32to8Kern<<<1, 5>>>(gpuMemInput, gpuMemOutput);
    float8_s * cpuMemOutput = new float8_s[5]; //fuck memory free
    CUDA_CHECK(cudaMemcpy(cpuMemOutput, gpuMemOutput, 5*sizeof(float8_s), cudaMemcpyDefault));
    float * output = new float[5];

    for (int i = 0; i<5; i++) {
    	output[i] = float8to32(cpuMemOutput[i]);
    	std::cout << "32 bit: " << base_vector[i] << " 8 bit: " << output[i] << std::endl;
    }
    delete[] cpuMemOutput;
    delete[] output;

    //BIG TEST
    std::vector<float> base_vector2;
    base_vector2.resize(122500000);
    float * gpuMemInput2;
    float8_s * gpuMemOutput2;
    float * output_gpu;

    float * output_final = new float[122500000];

    float8_s * cpuMemOutput2 = new float8_s[122500000];//LEAKYING

    for (auto &float_num : base_vector2) {
    	//float_num = 2.3123f;
    	float_num = (rand() % 800)/100.0f + 0.321451f;
    }


    std::cout << "MALLOC_DONE" << std::endl;
    CUDA_CHECK(cudaMalloc(&gpuMemInput2, base_vector2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_gpu, base_vector2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpuMemOutput2, base_vector2.size()*sizeof(float8_s)));
    CUDA_CHECK(cudaMemcpy(gpuMemInput2, base_vector2.data(), base_vector2.size()*sizeof(float), cudaMemcpyDefault));
    float32to8Kern<<<122500, 1000>>>(gpuMemInput2, gpuMemOutput2);
    //CUDA_CHECK(cudaMemcpy(cpuMemOutput2, gpuMemOutput2, base_vector2.size()*sizeof(float8_s), cudaMemcpyDefault));
    float8to32Kern<<<122500, 1000>>>(gpuMemOutput2, output_gpu);
    CUDA_CHECK(cudaMemcpy(output_final, output_gpu, base_vector2.size()*sizeof(float8_s), cudaMemcpyDefault));

    for (int i = 0; i<5; i++) {
    	std::cout << "32 bit: " << base_vector2[i + 3123231] << " 8 bit: " << output_final[i + 3123231] << std::endl;
    }
    delete[] cpuMemOutput2;
    delete[] output_final;
    cudaFree(gpuMemOutput2);
    cudaFree(gpuMemInput2);
    cudaFree(output_gpu);

    //FLOAT 16 test
    std::cout << "FLOAT 16 test with nvidia __half" << std::endl;
    float * gpuMemInput3;
    __half * gpuMemOutput3;
    float * output_gpu3;

    float * output_final3 = new float[122500000];

    __half * cpuMemOutput3 = new __half[122500000];//LEAKYING

    std::cout << "MALLOC_DONE" << std::endl;
    CUDA_CHECK(cudaMalloc(&gpuMemInput3, base_vector2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_gpu3, base_vector2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpuMemOutput3, base_vector2.size()*sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(gpuMemInput3, base_vector2.data(), base_vector2.size()*sizeof(float), cudaMemcpyDefault));
    float32to16Kern<<<122500, 1000>>>(gpuMemInput3, gpuMemOutput3);
    //CUDA_CHECK(cudaMemcpy(cpuMemOutput2, gpuMemOutput2, base_vector2.size()*sizeof(float8_s), cudaMemcpyDefault));
    float16to32Kern<<<122500, 1000>>>(gpuMemOutput3, output_gpu3);
    CUDA_CHECK(cudaMemcpy(output_final3, output_gpu3, base_vector2.size()*sizeof(__half), cudaMemcpyDefault));

    for (int i = 0; i<5; i++) {
    	std::cout << "32 bit: " << base_vector2[i + 3123231] << " 16 bit: " << output_final3[i + 3123231] << std::endl;
    }
    delete[] cpuMemOutput3;
    delete[] output_final3;
    cudaFree(gpuMemOutput3);
    cudaFree(gpuMemInput3);
    cudaFree(output_gpu3);

    //FLOAT 16 test
    std::cout << "FLOAT 16 test with handwritten implemented float16" << std::endl;
    float * gpuMemInput4;
    float16_s * gpuMemOutput4;
    float * output_gpu4;

    float * output_final4 = new float[122500000];

    float16_s * cpuMemOutput4 = new float16_s[122500000];//LEAKYING

    std::cout << "MALLOC_DONE" << std::endl;
    CUDA_CHECK(cudaMalloc(&gpuMemInput4, base_vector2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_gpu4, base_vector2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpuMemOutput4, base_vector2.size()*sizeof(float16_s)));
    CUDA_CHECK(cudaMemcpy(gpuMemInput4, base_vector2.data(), base_vector2.size()*sizeof(float), cudaMemcpyDefault));
    float32to16Kern<<<122500, 1000>>>(gpuMemInput4, gpuMemOutput4);
    //CUDA_CHECK(cudaMemcpy(cpuMemOutput2, gpuMemOutput2, base_vector2.size()*sizeof(float8_s), cudaMemcpyDefault));
    float16to32Kern<<<122500, 1000>>>(gpuMemOutput4, output_gpu4);
    CUDA_CHECK(cudaMemcpy(output_final4, output_gpu4, base_vector2.size()*sizeof(float16_s), cudaMemcpyDefault));

    for (int i = 0; i<5; i++) {
    	std::cout << "32 bit: " << base_vector2[i + 3123231] << " 16 bit: " << output_final4[i + 3123231] << std::endl;
    }
    delete[] cpuMemOutput3;
    delete[] output_final3;
    cudaFree(gpuMemOutput3);
    cudaFree(gpuMemInput3);
    cudaFree(output_gpu3);

    return 0;
}
