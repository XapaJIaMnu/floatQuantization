#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

float toFloat(uint8_t x) {
    return x / 255.0e7;
}

uint8_t fromFloat(float x) {
    if (x < 0) return 0;
    if (x > 1e-7) return 255;
    return 255.0e7 * x; // this truncates; add 0.5 to round instead
}

int main() {
  const float a = 32.12314f;
  __half test2 = __float2half(a);
  float b = __half2float(test2);
  std::cout << a << std::endl;
  std::cout << b << std::endl;
  
  uint8_t small_num = fromFloat(a);
  float reconstructed = toFloat(small_num);
  std::cout << small_num << std::endl;
  std::cout << reconstructed << std::endl;
  return 0;
}
