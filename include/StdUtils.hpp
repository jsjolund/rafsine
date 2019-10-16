#pragma once

#include <thrust/device_vector.h>
#include <algorithm>
#include <iostream>

#include "Eigen/Geometry"

#define NaN std::numeric_limits<real>::quiet_NaN()

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

inline void hash_combine(std::size_t* seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t* seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  hash_combine(seed, rest...);
}

namespace std {
template <>
struct hash<Eigen::Vector3i> {
  std::size_t operator()(const Eigen::Vector3i& p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(&seed, p.x(), p.y(), p.z());
    return seed;
  }
};
}  // namespace std

template <class T>
std::ostream& operator<<(std::ostream& os, thrust::device_vector<T> v) {
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<float>(os, ", "));
  return os;
}

inline std::ostream& operator<<(std::ostream& os, Eigen::Vector3i v) {
  os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, Eigen::Vector3f v) {
  os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return os;
}
