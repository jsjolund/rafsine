
#include <gtest/gtest.h>

#include "DdQq.hpp"

TEST(DdQq, Ranks) {
  // Positive x-axis
  for (int i = 0; i < 9; i++) ASSERT_EQ(D3Q27vectors[D3Q27ranks[1][i]].x(), 1);
  // Negative x-axis
  for (int i = 0; i < 9; i++) ASSERT_EQ(D3Q27vectors[D3Q27ranks[2][i]].x(), -1);
  // Positive y-axis
  for (int i = 0; i < 9; i++) ASSERT_EQ(D3Q27vectors[D3Q27ranks[3][i]].y(), 1);
  // Negative y-axis
  for (int i = 0; i < 9; i++) ASSERT_EQ(D3Q27vectors[D3Q27ranks[4][i]].y(), -1);
  // Positive z-axis
  for (int i = 0; i < 9; i++) ASSERT_EQ(D3Q27vectors[D3Q27ranks[5][i]].z(), 1);
  // Negative z-axis
  for (int i = 0; i < 9; i++) ASSERT_EQ(D3Q27vectors[D3Q27ranks[6][i]].z(), -1);
}

TEST(DdQq, Opposite) {
  unsigned int D3Q27OppositeHost[27] = {
    0,  2,  1,  4,  3,  6,  5,  8,  7,  10, 9,  12, 11, 14,
    13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25};
  for (int i = 0; i < 27; i++) {
    Vector3<int> v = D3Q27vectors[i];
    Vector3<int> u = D3Q27vectors[D3Q27OppositeHost[i]];
    ASSERT_EQ(v, -u);
  }
}
