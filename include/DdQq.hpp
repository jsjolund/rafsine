#pragma once

#include <glm/vec3.hpp>
#include "CudaUtils.hpp"

const glm::ivec3 D3Q27[27] = {
    // Origin
    glm::ivec3(0, 0, 0),
    // 6 faces
    glm::ivec3(1, 0, 0),
    glm::ivec3(-1, 0, 0),
    glm::ivec3(0, 1, 0),
    glm::ivec3(0, -1, 0),
    glm::ivec3(0, 0, 1),
    glm::ivec3(0, 0, -1),
    // 12 edges
    glm::ivec3(1, 1, 0),
    glm::ivec3(-1, -1, 0),
    glm::ivec3(1, -1, 0),
    glm::ivec3(-1, 1, 0),
    glm::ivec3(1, 0, 1),
    glm::ivec3(-1, 0, -1),
    glm::ivec3(1, 0, -1),
    glm::ivec3(-1, 0, 1),
    glm::ivec3(0, 1, 1),
    glm::ivec3(0, -1, -1),
    glm::ivec3(0, 1, -1),
    glm::ivec3(0, -1, 1),
    // 8 corners
    glm::ivec3(1, 1, 1),
    glm::ivec3(-1, -1, -1),
    glm::ivec3(-1, 1, 1),
    glm::ivec3(1, -1, -1),
    glm::ivec3(1, -1, 1),
    glm::ivec3(-1, 1, -1),
    glm::ivec3(1, 1, -1),
    glm::ivec3(-1, -1, 1),
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
