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

/**
 * @brief D3Q27 lattice directions. Don't change the order!
 *
 */
const Eigen::Vector3i D3Q27[27] = {
    // Origin
    Eigen::Vector3i(0, 0, 0),  // 0
    // 6 faces
    Eigen::Vector3i(1, 0, 0),   // 1
    Eigen::Vector3i(-1, 0, 0),  // 2
    Eigen::Vector3i(0, 1, 0),   // 3
    Eigen::Vector3i(0, -1, 0),  // 4
    Eigen::Vector3i(0, 0, 1),   // 5
    Eigen::Vector3i(0, 0, -1),  // 6
    // 12 edges
    Eigen::Vector3i(1, 1, 0),    // 7
    Eigen::Vector3i(-1, -1, 0),  // 8
    Eigen::Vector3i(1, -1, 0),   // 9
    Eigen::Vector3i(-1, 1, 0),   // 10
    Eigen::Vector3i(1, 0, 1),    // 11
    Eigen::Vector3i(-1, 0, -1),  // 12
    Eigen::Vector3i(1, 0, -1),   // 13
    Eigen::Vector3i(-1, 0, 1),   // 14
    Eigen::Vector3i(0, 1, 1),    // 15
    Eigen::Vector3i(0, -1, -1),  // 16
    Eigen::Vector3i(0, 1, -1),   // 17
    Eigen::Vector3i(0, -1, 1),   // 18
    // 8 corners
    Eigen::Vector3i(1, 1, 1),     // 19
    Eigen::Vector3i(-1, -1, -1),  // 20
    Eigen::Vector3i(-1, 1, 1),    // 21
    Eigen::Vector3i(1, -1, -1),   // 22
    Eigen::Vector3i(1, -1, 1),    // 23
    Eigen::Vector3i(-1, 1, -1),   // 24
    Eigen::Vector3i(1, 1, -1),    // 25
    Eigen::Vector3i(-1, -1, 1),   // 26
};

extern __constant__ real D3Q27directions[27 * 3];

extern __constant__ int D3Q27directionsOpposite[27];

extern const int D3Q27ranks[7][9];

extern __constant__ real D3Q19weights[19];

extern __constant__ real D3Q7weights[7];
