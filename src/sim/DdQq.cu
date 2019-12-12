#include "DdQq.hpp"

const Eigen::Vector3i D3Q27vectors[27] = {
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

const int D3Q27ranks[7][9] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},          // padding
    {1, 7, 9, 11, 13, 19, 22, 23, 25},    // positive x-axis
    {2, 8, 10, 12, 14, 20, 21, 24, 26},   // negative x-axis
    {3, 7, 10, 15, 17, 19, 21, 24, 25},   // positive y-axis
    {4, 8, 9, 16, 18, 20, 22, 23, 26},    // negative y-axis
    {5, 11, 14, 15, 18, 19, 21, 23, 26},  // positive z-axis
    {6, 12, 13, 16, 17, 20, 22, 24, 25}   // negative z-axis
};

__constant__ real D3Q27[81] = {
    // Origin
    0, 0, 0,
    // 6 faces
    1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1,
    // 12 edges
    1, 1, 0, -1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0,
    1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, -1, 1,
    // 8 corners
    1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1,
    -1, 1};

__constant__ int D3Q27Opposite[27] = {0,  2,  1,  4,  3,  6,  5,  8,  7,
                                      10, 9,  12, 11, 14, 13, 16, 15, 18,
                                      17, 20, 19, 22, 21, 24, 23, 26, 25};

__constant__ real D3Q19weights[19] = {
    1.0 / 3.0,  1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

__constant__ real D3Q7weights[7] = {1.0 / 4.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0,
                                    1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0};
