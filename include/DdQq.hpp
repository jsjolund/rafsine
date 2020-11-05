#pragma once

#include "CudaUtils.hpp"
#include "Vector3.hpp"

namespace D3Q4 {
enum Enum { ORIGIN, X_AXIS, Y_AXIS, Z_AXIS };
}  // namespace D3Q4

namespace D3Q7 {
enum Enum {
  ORIGIN,
  X_AXIS_POS,
  X_AXIS_NEG,
  Y_AXIS_POS,
  Y_AXIS_NEG,
  Z_AXIS_POS,
  Z_AXIS_NEG
};
}  // namespace D3Q7

extern const Vector3<int> D3Q27vectors[27];

extern const unsigned int D3Q27ranks[7][9];

extern __constant__ real_t D3Q27[81];

extern __constant__ unsigned int D3Q27Opposite[27];

extern __constant__ real_t D3Q19weights[19];

extern __constant__ real_t D3Q7weights[7];
