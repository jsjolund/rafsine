#pragma once

#include <thrust/device_vector.h>
#include <algorithm>
#include <iostream>

#include "Vector3.hpp"

#define NaN std::numeric_limits<real_t>::quiet_NaN()

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

inline void hash_combine(std::size_t*) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t* seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  hash_combine(seed, rest...);
}

namespace std {
template <>
struct hash<Vector3<int>> {
  std::size_t operator()(const Vector3<int>& p) const {
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

template <class T>
std::ostream& operator<<(std::ostream& os, thrust::host_vector<T> v) {
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<float>(os, ", "));
  return os;
}

inline std::ostream& operator<<(std::ostream& os, Vector3<int> v) {
  os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, Vector3<real_t> v) {
  os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return os;
}
