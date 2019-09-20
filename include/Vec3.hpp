#pragma once

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <glm/vec3.hpp>

#include "CudaUtils.hpp"

#define NaN std::numeric_limits<real>::quiet_NaN()

inline void hash_combine(std::size_t *seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t *seed, const T &v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  hash_combine(seed, rest...);
}

inline std::ostream &operator<<(std::ostream &os, glm::ivec3 v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, glm::vec3 v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}

namespace std {
template <>
struct hash<glm::ivec3> {
  std::size_t operator()(const glm::ivec3 &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(&seed, p.x, p.y, p.z);
    return seed;
  }
};
}  // namespace std

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

/// Compute the absolute value of a
template <class T>
inline T abs(const T &a) {
  return (a > 0) ? a : (-a);
}

/// Compute the minimum of a and b
template <class T>
inline const T &min(const T &a, const T &b) {
  return (a < b) ? a : b;
}

/// Compute the maximum of a and b
template <class T>
inline const T &max(const T &a, const T &b) {
  return (a > b) ? a : b;
}

template <typename T>
struct vec3;

/// Structure to regroup 3 numbers of type T. (useful for 3D coordinates)
template <typename T>
struct vec3 {
  static const bool empty = false;
  /// number of components
  static const int dimension = 3;
  /// define the null vector
  static const vec3 ZERO;
  /// define the base unit vector along x-axis
  static const vec3 X;
  /// define the base unit vector along y-axis
  static const vec3 Y;
  /// define the base unit vector along z-axis
  static const vec3 Z;
  /// component of the vector along x-axis
  T x;
  /// component of the vector along y-axis
  T y;
  /// component of the vector along z-axis
  T z;
  /// Default constructor
  vec3() {
    x = 0;
    y = 0;
    z = 0;
  }
  /// Constructor
  vec3(T x, T y, T z) {
    this->x = x;
    this->y = y;
    this->z = z;
  }
  /// Constructor with another vec3
  template <typename U>
  explicit vec3(vec3<U> v) {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
  }

  template <typename U>
  operator vec3<U>() {
    return vec3<U>(static_cast<U>(x), static_cast<U>(y), static_cast<U>(z));
  }

  // compute the norm
  inline T norm() const { return sqrtf((*this) * (*this)); }
  // normalise the vector (divide by its norm)
  inline void normalize() {
    T norm = sqrtf((*this) * (*this));
    if (norm == 0) {
      throw std::invalid_argument("Vector is zero");
    } else {
      x /= norm;
      y /= norm;
      z /= norm;
    }
  }

  /// Output a vector
  template <typename U>
  friend std::ostream &operator<<(std::ostream &out, const vec3<U> &v);
  /// Normalise a vector
  template <typename U>
  static inline vec3<U> normalize(const vec3<U> &v) {
    U norm = v.norm();
    if (norm == 0) throw std::invalid_argument("Vector is zero");
    return vec3<U>(v.x / norm, v.y / norm, v.z / norm);
  }
};
template <typename T>
const vec3<T> vec3<T>::ZERO = vec3<T>(0, 0, 0);
template <typename T>
const vec3<T> vec3<T>::X = vec3<T>(1, 0, 0);
template <typename T>
const vec3<T> vec3<T>::Y = vec3<T>(0, 1, 0);
template <typename T>
const vec3<T> vec3<T>::Z = vec3<T>(0, 0, 1);

/// 3D vector of real
typedef vec3<real> vec3r;
/// 3D vector of unsigned int
typedef vec3<unsigned int> vec3ui;

/// reverse a vector
template <typename T>
inline vec3<T> operator-(const vec3<T> &v) {
  return vec3<T>(-v.x, -v.y, -v.z);
}
/// Add two vectors together.
template <typename T>
inline vec3<T> operator+(const vec3<T> &v1, const vec3<T> &v2) {
  return vec3<T>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
/// Add two vectors together.
template <typename T, typename U>
inline vec3r operator+(const vec3<T> &v1, const vec3<U> &v2) {
  return vec3r(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
/// Add a vector to the current vector.
template <typename T>
inline void operator+=(vec3<T> &v1, const vec3<T> &v2) {
  v1.x += v2.x;
  v1.y += v2.y;
  v1.z += v2.z;
}
/// Subtract two vectors together.
template <typename T>
inline vec3<T> operator-(const vec3<T> &v1, const vec3<T> &v2) {
  return vec3<T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
/// Subtract a vector to the current vector.
template <typename T>
inline void operator-=(vec3<T> &v1, const vec3<T> &v2) {
  v1.x -= v2.x;
  v1.y -= v2.y;
  v1.z -= v2.z;
}
/// Compute the scalar product of two vectors.
template <typename T>
inline T operator*(const vec3<T> &v1, const vec3<T> &v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
/// Compute the cross product of two vectors.
template <typename T>
inline vec3<T> operator^(const vec3<T> &v1, const vec3<T> &v2) {
  return vec3<T>(v1.y * v2.z - v2.y * v1.z, v1.z * v2.x - v2.z * v1.x,
                 v1.x * v2.y - v2.x * v1.y);
}
/// Multiply a vector by a scalar
template <typename T1, typename T2>
inline vec3<T2> operator*(const T1 &a, const vec3<T2> &v) {
  return vec3<T2>(a * v.x, a * v.y, a * v.z);
}
/// Multiply a vector by a scalar
template <typename T1, typename T2>
inline void operator*=(vec3<T1> &v, const T2 &a) {
  v.x *= a;
  v.y *= a;
  v.z *= a;
}

/// Divide a vector by a scalar
template <typename T1, typename T2>
inline vec3<T2> operator/(const vec3<T2> &v, const T1 &a) {
  return vec3<T2>(v.x / a, v.y / a, v.z / a);
}
/// Divide a vector by a scalar
template <typename T1, typename T2>
inline void operator/=(vec3<T1> &v, const T2 &a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
}

/// Output a vector
template <typename T>
std::ostream &operator<<(std::ostream &out, const vec3<T> &v) {
  out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return out;
}
