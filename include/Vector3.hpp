#pragma once

#include <cmath>
#include <iostream>

#include "CudaUtils.hpp"

/**
 * @brief Template class for three element vectors.
 *
 * @tparam T Numeric type
 */
template <class T>
class Vector3 {
 public:
  /**
   * @return Zero vector
   */
  CUDA_CALLABLE_MEMBER inline Vector3() {
    _v[0] = 0.0;
    _v[1] = 0.0;
    _v[2] = 0.0;
  }
  /**
   * @brief Construct vector from values
   *
   * @param x
   * @param y
   * @param z
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3(const T x, const T y, const T z) {
    _v[0] = x;
    _v[1] = y;
    _v[2] = z;
  }
  /**
   * @brief Copy constructor
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3(const Vector3<T>& v) {
    _v[0] = v[0];
    _v[1] = v[1];
    _v[2] = v[2];
  }
  /**
   * @return Read only X component
   */
  CUDA_CALLABLE_MEMBER inline const T x() const { return _v[0]; }
  /**
   * @return Read only Y component
   */
  CUDA_CALLABLE_MEMBER inline const T y() const { return _v[1]; }
  /**
   * @return Read only Z component
   */
  CUDA_CALLABLE_MEMBER inline const T z() const { return _v[2]; }
  /**
   * @return Read/write X component
   */
  CUDA_CALLABLE_MEMBER inline T& x() { return _v[0]; }
  /**
   * @return Read/write Y component
   */
  CUDA_CALLABLE_MEMBER inline T& y() { return _v[1]; }
  /**
   * @return Read/write Z component
   */
  CUDA_CALLABLE_MEMBER inline T& z() { return _v[2]; }
  /**
   * @return L2 norm of vector
   */
  CUDA_CALLABLE_MEMBER const T norm() const {
    return (T)sqrt(dot(*this));  // cast to type
  }
  /**
   * @return Sum of vector components
   */
  CUDA_CALLABLE_MEMBER const T sum() const { return _v[0] + _v[1] + _v[2]; }
  /**
   * @brief Dot product of this vector and another
   *
   * @param v Other vector
   * @return Dot product
   */
  CUDA_CALLABLE_MEMBER inline const T dot(const Vector3<T>& v) const {
    return _v[0] * v[0] + _v[1] * v[1] + _v[2] * v[2];
  }
  /**
   * @brief Cross product of this vector and another
   *
   * @param v Other vector
   * @return Cross product
   */
  CUDA_CALLABLE_MEMBER inline const Vector3<T> cross(
      const Vector3<T>& v) const {
    return Vector3<T>((_v[1] * v[2]) - (_v[2] * v[1]),
                      (_v[2] * v[0]) - (_v[0] * v[2]),
                      (_v[0] * v[1]) - (_v[1] * v[0]));
  }
  /**
   * @brief Set this vector to zero
   *
   * @return This vector set to zero
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& zero() {
    _v[0] = 0.0;
    _v[1] = 0.0;
    _v[2] = 0.0;
    return *this;
  }
  /**
   * @brief Set the components of this vector
   *
   * @param x
   * @param y
   * @param z
   * @return This vector
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& set(const T x, const T y, const T z) {
    _v[0] = x;
    _v[1] = y;
    _v[2] = z;
    return *this;
  }
  /**
   * @brief Normalize this vector
   *
   * @return This vector normalized
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& normalize() {
    T n = norm();
    if (n) {
      _v[0] /= n;
      _v[1] /= n;
      _v[2] /= n;
    }
    return *this;
  }

  /**
   * @brief Assign values to this vector from another
   *
   * @param v Another vector
   * @return This vector
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator=(const Vector3<T>& v) {
    _v[0] = v[0];
    _v[1] = v[1];
    _v[2] = v[2];
    return *this;
  }
  /**
   * @brief Read only value at index in backing array
   *
   * @param i Index 0 to 2
   * @return The value
   */
  CUDA_CALLABLE_MEMBER inline const T operator[](const int i) const {
    return _v[i];
  }

  /**
   * @brief Read/write only value at index in backing array
   *
   * @param i Index 0 to 2
   * @return The value
   */
  CUDA_CALLABLE_MEMBER inline T& operator[](const int i) { return _v[i]; }

  /**
   * @brief Unary negate
   *
   * @return New vector
   */
  CUDA_CALLABLE_MEMBER inline const Vector3<T> operator-() {
    return Vector3<T>(-_v[0], -_v[1], -_v[2]);
  }

  /**
   * @brief Scalar addition
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator+=(const T v) {
    _v[0] += v;
    _v[1] += v;
    _v[2] += v;
    return *this;
  }
  /**
   * @brief Scalar subtraction
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator-=(const T v) {
    _v[0] -= v;
    _v[1] -= v;
    _v[2] -= v;
    return *this;
  }
  /**
   * @brief Scalar multiplication
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator*=(const T v) {
    _v[0] *= v;
    _v[1] *= v;
    _v[2] *= v;
    return *this;
  }
  /**
   * @brief Scalar division
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator/=(const T v) {
    _v[0] /= v;
    _v[1] /= v;
    _v[2] /= v;
    return *this;
  }

  /**
   * @brief Vector addition
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator+=(const Vector3<T>& v) {
    _v[0] += v[0];
    _v[1] += v[1];
    _v[2] += v[2];
    return *this;
  }
  /**
   * @brief Vector subtraction
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator-=(const Vector3<T>& v) {
    _v[0] -= v[0];
    _v[1] -= v[1];
    _v[2] -= v[2];
    return *this;
  }
  /**
   * @brief Element-wise multiplication
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator*=(const Vector3<T>& v) {
    _v[0] *= v[0];
    _v[1] *= v[1];
    _v[2] *= v[2];
    return *this;
  }
  /**
   * @brief Element-wise division
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline Vector3<T>& operator/=(const Vector3<T>& v) {
    _v[0] /= v[0];
    _v[1] /= v[1];
    _v[2] /= v[2];
    return *this;
  }

  /**
   * @brief Test vector for equality
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline bool operator==(const Vector3<T>& v) const {
    return _v[0] == v[0] && _v[1] == v[1] && _v[2] == v[2];
  }
  /**
   * @brief Test vector for inequality
   *
   * @param v
   * @return
   */
  CUDA_CALLABLE_MEMBER inline bool operator!=(const Vector3<T>& v) const {
    return _v[0] != v[0] || _v[1] != v[1] || _v[2] != v[2];
  }

  /**
   * @return T* Reference to array (use with caution)
   */
  T* ptr() { return _v; }

  /**
   * @brief Cast between vector types (must have same signedness)
   *
   * @tparam D
   * @return Vector3<D>
   */
  template <typename D>
  operator Vector3<D>() const {
    static_assert(
        std::is_same<D, size_t>::value || std::is_same<D, unsigned int>::value,
        "cannot cast types");
    return Vector3<D>(static_cast<D>(_v[0]), static_cast<D>(_v[1]),
                      static_cast<D>(_v[2]));
  }

 private:
  T _v[3];
};

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator&&(const Vector3<T>& v1,
                                                 const Vector3<T>& v2) {
  return Vector3<T>(v1[0] && v2[0], v1[1] && v2[1], v1[2] && v2[2]);
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator||(const Vector3<T>& v1,
                                                 const Vector3<T>& v2) {
  return Vector3<T>(v1[0] || v2[0], v1[1] || v2[1], v1[2] || v2[2]);
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator+(const Vector3<T>& v,
                                                const T& s) {
  return Vector3<T>(v) += s;
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator-(const Vector3<T>& v,
                                                const T& s) {
  return Vector3<T>(v) -= s;
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator*(const Vector3<T>& v,
                                                const T& s) {
  return Vector3<T>(v) *= s;
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator/(const Vector3<T>& v,
                                                const T& s) {
  return Vector3<T>(v) /= s;
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator+(const Vector3<T>& v1,
                                                const Vector3<T>& v2) {
  return Vector3<T>(v1) += v2;
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator-(const Vector3<T>& v1,
                                                const Vector3<T>& v2) {
  return Vector3<T>(v1) -= v2;
}

template <class T>
CUDA_CALLABLE_MEMBER const T operator*(const Vector3<T>& v1,
                                       const Vector3<T>& v2) {
  return v1.dot(v2);
}

template <class T>
CUDA_CALLABLE_MEMBER const Vector3<T> operator^(const Vector3<T>& v1,
                                                const Vector3<T>& v2) {
  return v1.cross(v2);
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Vector3<T> v) {
  os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return os;
}
