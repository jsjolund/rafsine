// Generated by lbm_gen_ddqq.py
#include "DdQq.hpp"

// Direction vectors for host usage
const Vector3<int> D3Q27vectors[27] = {
    // Origin
    Vector3<int>(0, 0, 0),  // 0
    // 6 faces
    Vector3<int>(1, 0, 0),   // 1
    Vector3<int>(-1, 0, 0),  // 2
    Vector3<int>(0, 1, 0),   // 3
    Vector3<int>(0, -1, 0),  // 4
    Vector3<int>(0, 0, 1),   // 5
    Vector3<int>(0, 0, -1),  // 6
    // 12 edges
    Vector3<int>(1, 1, 0),    // 7
    Vector3<int>(-1, 1, 0),   // 8
    Vector3<int>(1, -1, 0),   // 9
    Vector3<int>(-1, -1, 0),  // 10
    Vector3<int>(1, 0, 1),    // 11
    Vector3<int>(-1, 0, 1),   // 12
    Vector3<int>(1, 0, -1),   // 13
    Vector3<int>(-1, 0, -1),  // 14
    Vector3<int>(0, 1, 1),    // 15
    Vector3<int>(0, -1, 1),   // 16
    Vector3<int>(0, 1, -1),   // 17
    Vector3<int>(0, -1, -1),  // 18
    // 8 corners
    Vector3<int>(1, 1, 1),    // 19
    Vector3<int>(-1, 1, 1),   // 20
    Vector3<int>(1, -1, 1),   // 21
    Vector3<int>(-1, -1, 1),  // 22
    Vector3<int>(1, 1, -1),   // 23
    Vector3<int>(-1, 1, -1),  // 24
    Vector3<int>(1, -1, -1),  // 25
    Vector3<int>(-1, -1, -1)  // 26
};

// Direction vectors for CUDA usage
__constant__ int D3Q27[81] = {
    // Origin
    0, 0, 0,  // 0
    // 6 faces
    1, 0, 0,   // 1
    -1, 0, 0,  // 2
    0, 1, 0,   // 3
    0, -1, 0,  // 4
    0, 0, 1,   // 5
    0, 0, -1,  // 6
    // 12 edges
    1, 1, 0,    // 7
    -1, 1, 0,   // 8
    1, -1, 0,   // 9
    -1, -1, 0,  // 10
    1, 0, 1,    // 11
    -1, 0, 1,   // 12
    1, 0, -1,   // 13
    -1, 0, -1,  // 14
    0, 1, 1,    // 15
    0, -1, 1,   // 16
    0, 1, -1,   // 17
    0, -1, -1,  // 18
    // 8 corners
    1, 1, 1,    // 19
    -1, 1, 1,   // 20
    1, -1, 1,   // 21
    -1, -1, 1,  // 22
    1, 1, -1,   // 23
    -1, 1, -1,  // 24
    1, -1, -1,  // 25
    -1, -1, -1  // 26
};

// Vector index table for ghost layer exchange ordering
const unsigned int D3Q27ranks[7][9] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},          // padding
    {1, 7, 9, 11, 13, 19, 21, 23, 25},    // positive x-axis
    {2, 8, 10, 12, 14, 20, 22, 24, 26},   // negative x-axis
    {3, 7, 8, 15, 17, 19, 20, 23, 24},    // positive y-axis
    {4, 9, 10, 16, 18, 21, 22, 25, 26},   // negative y-axis
    {5, 11, 12, 15, 16, 19, 20, 21, 22},  // positive z-axis
    {6, 13, 14, 17, 18, 23, 24, 25, 26}   // negative z-axis
};

// Opposing vector ordering
__constant__ unsigned int D3Q27Opposite[27] = {
    0,  2,  1,  4,  3,  6,  5,  10, 9,  8,  7,  14, 13, 12,
    11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19};

// D3Q27 lattice weights
__constant__ real_t D3Q27weights[27] = {
    0.296296296296296,   0.0740740740740741,  0.0740740740740741,
    0.0740740740740741,  0.0740740740740741,  0.0740740740740741,
    0.0740740740740741,  0.0185185185185185,  0.0185185185185185,
    0.0185185185185185,  0.0185185185185185,  0.0185185185185185,
    0.0185185185185185,  0.0185185185185185,  0.0185185185185185,
    0.0185185185185185,  0.0185185185185185,  0.0185185185185185,
    0.0185185185185185,  0.00462962962962963, 0.00462962962962963,
    0.00462962962962963, 0.00462962962962963, 0.00462962962962963,
    0.00462962962962963, 0.00462962962962963, 0.00462962962962963};

// D3Q19 lattice weights
__constant__ real_t D3Q19weights[19] = {
    0.333333333333333,  0.0555555555555556, 0.0555555555555556,
    0.0555555555555556, 0.0555555555555556, 0.0555555555555556,
    0.0555555555555556, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778, 0.0277777777777778, 0.0277777777777778,
    0.0277777777777778};

// D3Q7 lattice weights
__constant__ real_t D3Q7weights[7] = {0.0,
                                      0.166666666666667,
                                      0.166666666666667,
                                      0.166666666666667,
                                      0.166666666666667,
                                      0.166666666666667,
                                      0.166666666666667};
