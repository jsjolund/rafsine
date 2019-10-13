#pragma once

#include "CudaUtils.hpp"
#include "Eigen/Geometry"

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

extern const Eigen::Vector3i D3Q27[27];

extern const int D3Q27ranks[7][9];

extern __constant__ real D3Q27directions[81];

extern __constant__ int D3Q27directionsOpposite[27];

extern __constant__ real D3Q19weights[19];

extern __constant__ real D3Q7weights[7];
