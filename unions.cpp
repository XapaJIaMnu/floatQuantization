#include <iostream>

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
 
float16_s float32to16(float x) {
  float32_s f32={x}; // c99
  float16_s f16;

  // to 16
  f16.bits.s=f32.bits.s;
  f16.bits.e=std::max(-15,std::min(16,(int)(f32.bits.e-127))) +15;
  f16.bits.m=f32.bits.m >> 13;
  return f16;
 }
 
 float float16to32(float16_s f16) {
  // back to 32
  float32_s f32;
  f32.bits.s=f16.bits.s;
  f32.bits.e=(f16.bits.e-15)+127; // safe in this direction
  f32.bits.m=((uint32_t)f16.bits.m) << 13;
 
  return f32.v;
 }
 
float8_s float32to8(float x) {
  float32_s f32={x}; // c99
  float8_s f8;

  // to 8
  f8.bits.s=f32.bits.s;
  f8.bits.e=std::max(-3,std::min(4,(int)(f32.bits.e-127))) +3; //WRONG
  f8.bits.m=f32.bits.m >> 19;
  return f8;
 }
 
float float8to32(float8_s f8) {
  // back to 32
  float32_s f32;
  f32.bits.s=f8.bits.s;
  f32.bits.e=(f8.bits.e-3)+127; // safe in this direction
  f32.bits.m=((uint32_t)f8.bits.m) << 19;
 
  return f32.v;
 }

int main() {
    float a = -0.423243242f;
    float16_s quantized = float32to16(a);
    float b = float16to32(quantized);
    float8_s quantized8 = float32to8(a);
    float c = float8to32(quantized8);
    std::cout << "Size of 32 " << sizeof(a) << " Size of 16: " << sizeof(quantized) << " size of 8: " << sizeof(quantized8) << std::endl;
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 0.423243242f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 12.62415142f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 8.8798271f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 4.6991462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 2.23191462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 1.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = 0.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -12.62415142f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -8.8798271f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -4.6991462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -2.23191462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -1.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -0.77823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -0.23823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -0.15823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -0.13823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    a = -0.38823462f;
    quantized = float32to16(a);
    b = float16to32(quantized);
    quantized8 = float32to8(a);
    c = float8to32(quantized8);
    std::cout << "Float a: " << a << " float b: " << b << " float c: " << c << std::endl;
    
    return 0;

}
