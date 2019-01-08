#pragma once

#include <glm/vec3.hpp>
#include "CudaUtils.hpp"

const glm::ivec3 D3Q27[27] = {
    // Origin
    glm::ivec3(0, 0, 0),  // 0
    // 6 faces
    glm::ivec3(1, 0, 0),   // 1
    glm::ivec3(-1, 0, 0),  // 2
    glm::ivec3(0, 1, 0),   // 3
    glm::ivec3(0, -1, 0),  // 4
    glm::ivec3(0, 0, 1),   // 5
    glm::ivec3(0, 0, -1),  // 6
    // 12 edges
    glm::ivec3(1, 1, 0),    // 7
    glm::ivec3(-1, -1, 0),  // 8
    glm::ivec3(1, -1, 0),   // 9
    glm::ivec3(-1, 1, 0),   // 10
    glm::ivec3(1, 0, 1),    // 11
    glm::ivec3(-1, 0, -1),  // 12
    glm::ivec3(1, 0, -1),   // 13
    glm::ivec3(-1, 0, 1),   // 14
    glm::ivec3(0, 1, 1),    // 15
    glm::ivec3(0, -1, -1),  // 16
    glm::ivec3(0, 1, -1),   // 17
    glm::ivec3(0, -1, 1),   // 18
    // 8 corners
    glm::ivec3(1, 1, 1),     // 19
    glm::ivec3(-1, -1, -1),  // 20
    glm::ivec3(-1, 1, 1),    // 21
    glm::ivec3(1, -1, -1),   // 22
    glm::ivec3(1, -1, 1),    // 23
    glm::ivec3(-1, 1, -1),   // 24
    glm::ivec3(1, 1, -1),    // 25
    glm::ivec3(-1, -1, 1),   // 26
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

const int D3Q27directionsOpposite[27] = {0,  2,  1,  4,  3,  6,  5,  8,  7,
                                         10, 9,  12, 11, 14, 13, 16, 15, 18,
                                         17, 20, 19, 22, 21, 24, 23, 26, 25};

__constant__ real D3Q27directions[27 * 3] = {
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
__constant__ real D3Q19directions[19 * 3] = {
    0, 0, 0,
    // 6 faces
    1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1,
    // Diagonal
    1, 1, 0, -1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0,
    1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, -1, 1};
__constant__ int D3Q19directionsOpposite[19] = {
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
__constant__ real D3Q19weights[19] = {
    1.0f / 3.0f,  1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

__constant__ real D3Q7directions[7 * 3] = {0, 0, 0,
                                           // 6 faces
                                           1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0,
                                           0, 0, 1, 0, 0, -1};
__constant__ int D3Q7directionsOpposite[7] = {0, 2, 1, 4, 3, 6, 5};
__constant__ real D3Q7weights[7] = {0,           1.0f / 6.0f, 1.0f / 6.0f,
                                    1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                    1.0f / 6.0f};
