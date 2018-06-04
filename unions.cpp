#include <iostream>
#include "floatUnion.h"
#include <memory>
#ifdef USE_MPI
#include <mpi.h>
#endif

void mpiTest(int argc, char ** argv) {
    #ifdef USE_MPI
    //MPI_TEST
    std::vector<float> base_vector{-4.6991462f, 8.8798271f, -0.38823462f, 0.13823462f, -1.16823462f};
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    int mpi_my_rank_;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_my_rank_);

    std::vector<float> from_8_MPI;
    from_8_MPI.resize(5);

    std::vector<float> from_16_MPI;
    from_16_MPI.resize(5);

    std::vector<float8_s> float8_container_MPI;
    float8_container_MPI.resize(5);

    std::vector<float16_s> float16_container_MPI;
    float16_container_MPI.resize(5);
    if (mpi_my_rank_ == 0) {
        float32to8Vec(base_vector, float8_container_MPI);
        float32to16Vec(base_vector, float16_container_MPI);
    }

    int bcast_result = MPI_Bcast(float16_container_MPI.data(), //This is now the updated params.
            5,
            MPI_SHORT,
            0, //Root process
            MPI_COMM_WORLD);
    if (bcast_result != MPI_SUCCESS) {
        std::cout << "bcast failed." << std::endl;
    }
    bcast_result = MPI_Bcast(float8_container_MPI.data(), //This is now the updated params.
            5,
            MPI_BYTE,
            0, //Root process
            MPI_COMM_WORLD);
    if (bcast_result != MPI_SUCCESS) {
        std::cout << "bcast failed." << std::endl;
    }

    //Convert back
    float8to32Vec(float8_container_MPI, from_8_MPI);
    float16to32Vec(float16_container_MPI, from_16_MPI);

    //PRINT

    for (size_t i = 0; i < base_vector.size(); i++) {
        std::cout << "Float 32: " << base_vector[i] << " Float 16: " << from_16_MPI[i] << " Float 8: " << from_8_MPI[i] << std::endl;
    }
    MPI_Finalize();
    #endif
}


void vectorTest() {
    std::vector<float8_s> float8_container;
    float8_container.resize(5);

    std::vector<float16_s> float16_container;
    float16_container.resize(5);

    std::vector<float> base_vector{-4.6991462f, 8.8798271f, -0.38823462f, 0.13823462f, -1.16823462f};

    std::vector<float> from_8;
    from_8.resize(5);

    std::vector<float> from_16;
    from_16.resize(5);

    float32to8Vec(base_vector, float8_container);
    float32to16Vec(base_vector, float16_container);

    float8to32Vec(float8_container, from_8);
    float16to32Vec(float16_container, from_16);

    for (size_t i = 0; i < base_vector.size(); i++) {
        std::cout << "Float 32: " << base_vector[i] << " Float 16: " << from_16[i] << " Float 8: " << from_8[i] << std::endl;
    }
}

void sizeTest() {
	std::cout << "SIZE TEST" << std::endl;
    std::vector<float8_s> float8_container;
    float8_container.resize(122500000);

    std::vector<float16_s> float16_container;
    float16_container.resize(122500000);

    std::vector<float> base_vector;
    base_vector.resize(122500000);

    for (auto float_num : base_vector) {
    	float_num = 2.3123f;
    }

    std::vector<float> from_8;
    from_8.resize(122500000);

    std::vector<float> from_16;
    from_16.resize(122500000);
    std::cout << "ALLOCATION_DONE" << std::endl;

    float32to8Vec(base_vector, float8_container);
    float32to16Vec(base_vector, float16_container);

    float8to32Vec(float8_container, from_8);
    float16to32Vec(float16_container, from_16);
    std::cout << "SIZE TEST DONE" << std::endl;
    //for (size_t i = 0; i < base_vector.size(); i++) {
    //    std::cout << "Float 32: " << base_vector[i] << " Float 16: " << from_16[i] << " Float 8: " << from_8[i] << std::endl;
    //}
}

void arrTest() {
	std::unique_ptr<float8_s[]> float8_container(new float8_s[5]);

    std::unique_ptr<float16_s[]> float16_container(new float16_s[5]);

    float base_array[5] = {-4.6991462f, 8.8798271f, -0.38823462f, 0.13823462f, -1.16823462f};

	std::unique_ptr<float> from_8(new float[5]);

    std::unique_ptr<float> from_16(new float[5]);

    float32to8Arr(base_array, float8_container.get(), 5);
    float32to16Arr(base_array, float16_container.get(), 5);

    float8to32Arr(float8_container.get(), from_8.get(), 5);
    float16to32Arr(float16_container.get(), from_16.get(), 5);

    for (size_t i = 0; i < 5; i++) {
        std::cout << "Float 32: " << base_array[i] << " Float 16: " << from_16.get()[i] << " Float 8: " << from_8.get()[i] << std::endl;
    }
}

void individualTest() {
        float a = -0.423243242f;
    float16_s quantized = float32to16(a);
    float b = float16to32(quantized);
    float8_s quantized8 = float32to8(a);
    float c = float8to32(quantized8);
    std::cout << "Size of 32 " << sizeof(a) << " Size of 16: " << sizeof(quantized) << " size of 8: " << sizeof(quantized8) << std::endl;
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 0.423243242f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 12.62415142f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 8.8798271f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 4.6991462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 2.23191462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 1.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = 0.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -12.62415142f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -8.8798271f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -4.6991462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -2.23191462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -1.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -0.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -0.23823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -0.15823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -0.13823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
    
    a = -0.38823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float 32: " << a << " Float 16: " << b << " Float 8: " << c << std::endl;
}

int main(int argc, char ** argv) {
    std::cout << "INDIVIDUAL TEST" << std::endl;
    individualTest();
    std::cout << "VECTOR TEST" << std::endl;
    vectorTest();
    std::cout << "ARRAY TEST" << std::endl;
    arrTest();
    std::cout << "MPI TEST" << std::endl;
    sizeTest();
    mpiTest(argc, argv);

    return 0;

}
