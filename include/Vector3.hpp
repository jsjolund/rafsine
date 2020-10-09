#ifndef VECTOR3_
#define VECTOR3_

#include <cmath>
#include <iostream>

#include "CudaUtils.hpp"

///
/// Template class for three element vectors.
///
template <class T>
class vector3 {
 public:
  CUDA_CALLABLE_MEMBER inline vector3() {
    _v[0] = 0.0;
    _v[1] = 0.0;
    _v[2] = 0.0;
  }
  CUDA_CALLABLE_MEMBER inline vector3(const T x, const T y, const T z) {
    _v[0] = x;
    _v[1] = y;
    _v[2] = z;
  }
  CUDA_CALLABLE_MEMBER inline vector3(const vector3<T>& v) {
    _v[0] = v[0];
    _v[1] = v[1];
    _v[2] = v[2];
  }

  CUDA_CALLABLE_MEMBER inline const T x() const { return _v[0]; }
  CUDA_CALLABLE_MEMBER inline const T y() const { return _v[1]; }
  CUDA_CALLABLE_MEMBER inline const T z() const { return _v[2]; }

  CUDA_CALLABLE_MEMBER inline T& x() { return _v[0]; }
  CUDA_CALLABLE_MEMBER inline T& y() { return _v[1]; }
  CUDA_CALLABLE_MEMBER inline T& z() { return _v[2]; }

  // math operations
  CUDA_CALLABLE_MEMBER const T norm() const {
    return (T)sqrt(dot(*this));  // cast to type
  }

  // const vector3<T> vector3<T>::abs() const {
  //   return vector3<T>(std::abs(_v[0]), std::abs(_v[1]), std::abs(_v[2]));
  // }

  CUDA_CALLABLE_MEMBER const T sum() const { return _v[0] + _v[1] + _v[2]; }

  CUDA_CALLABLE_MEMBER inline const T dot(const vector3<T>& v) const {
    return _v[0] * v[0] + _v[1] * v[1] + _v[2] * v[2];
  }

  CUDA_CALLABLE_MEMBER inline const vector3<T> cross(
      const vector3<T>& v) const {
    return vector3<T>((_v[1] * v[2]) - (_v[2] * v[1]),
                      (_v[2] * v[0]) - (_v[0] * v[2]),
                      (_v[0] * v[1]) - (_v[1] * v[0]));
  }

  // utility operations
  CUDA_CALLABLE_MEMBER inline vector3<T>& zero() {
    _v[0] = 0.0;
    _v[1] = 0.0;
    _v[2] = 0.0;
    return *this;
  }

  CUDA_CALLABLE_MEMBER inline vector3<T>& set(const T x, const T y, const T z) {
    _v[0] = x;
    _v[1] = y;
    _v[2] = z;
    return *this;
  }

  CUDA_CALLABLE_MEMBER inline vector3<T>& normalize() {
    T n = norm();
    if (n) {
      _v[0] /= n;
      _v[1] /= n;
      _v[2] /= n;
    }
    return *this;
  }

  // operators

  // assignment
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator=(const vector3<T>& v) {
    _v[0] = v[0];
    _v[1] = v[1];
    _v[2] = v[2];
    return *this;
  }

  // indexing
  CUDA_CALLABLE_MEMBER inline const T operator[](const int i) const {
    return _v[i];
  }
  // indexing
  CUDA_CALLABLE_MEMBER inline T& operator[](const int i) { return _v[i]; }

  // unary negate
  CUDA_CALLABLE_MEMBER inline const vector3<T> operator-() {
    return vector3<T>(-_v[0], -_v[1], -_v[2]);
  }

  // scalar addition
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator+=(const T v) {
    _v[0] += v;
    _v[1] += v;
    _v[2] += v;
    return *this;
  }
  // scalar subtraction
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator-=(const T v) {
    _v[0] -= v;
    _v[1] -= v;
    _v[2] -= v;
    return *this;
  }
  // scalar multiplication
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator*=(const T v) {
    _v[0] *= v;
    _v[1] *= v;
    _v[2] *= v;
    return *this;
  }
  // scalar division
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator/=(const T v) {
    _v[0] /= v;
    _v[1] /= v;
    _v[2] /= v;
    return *this;
  }

  // vector addition
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator+=(const vector3<T>& v) {
    _v[0] += v[0];
    _v[1] += v[1];
    _v[2] += v[2];
    return *this;
  }
  // vector subtraction
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator-=(const vector3<T>& v) {
    _v[0] -= v[0];
    _v[1] -= v[1];
    _v[2] -= v[2];
    return *this;
  }
  // element-wise multiplication
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator*=(const vector3<T>& v) {
    _v[0] *= v[0];
    _v[1] *= v[1];
    _v[2] *= v[2];
    return *this;
  }
  // element-wise division
  CUDA_CALLABLE_MEMBER inline vector3<T>& operator/=(const vector3<T>& v) {
    _v[0] /= v[0];
    _v[1] /= v[1];
    _v[2] /= v[2];
    return *this;
  }

  // test vector for equality
  CUDA_CALLABLE_MEMBER inline bool operator==(const vector3<T>& v) const {
    return _v[0] == v[0] && _v[1] == v[1] && _v[2] == v[2];
  }
  // test vector for inequality
  CUDA_CALLABLE_MEMBER inline bool operator!=(const vector3<T>& v) const {
    return _v[0] != v[0] || _v[1] != v[1] || _v[2] != v[2];
  }

  T* ptr() { return _v; }  // return reference to array (use with caution)

  template <typename D>
  operator vector3<D>() const {
    static_assert(
        std::is_same<D, size_t>::value || std::is_same<D, unsigned int>::value,
        "cannot cast types");
    return vector3<D>(static_cast<D>(_v[0]), static_cast<D>(_v[1]),
                      static_cast<D>(_v[2]));
  }

 private:
  T _v[3];
};

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator&&(const vector3<T>& v1,
                                                 const vector3<T>& v2) {
  return vector3<T>(v1[0] && v2[0], v1[1] && v2[1], v1[2] && v2[2]);
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator||(const vector3<T>& v1,
                                                 const vector3<T>& v2) {
  return vector3<T>(v1[0] || v2[0], v1[1] || v2[1], v1[2] || v2[2]);
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator+(const vector3<T>& v,
                                                const T& s) {
  return vector3<T>(v) += s;
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator-(const vector3<T>& v,
                                                const T& s) {
  return vector3<T>(v) -= s;
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator*(const vector3<T>& v,
                                                const T& s) {
  return vector3<T>(v) *= s;
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator/(const vector3<T>& v,
                                                const T& s) {
  return vector3<T>(v) /= s;
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator+(const vector3<T>& v1,
                                                const vector3<T>& v2) {
  return vector3<T>(v1) += v2;
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator-(const vector3<T>& v1,
                                                const vector3<T>& v2) {
  return vector3<T>(v1) -= v2;
}

template <class T>
CUDA_CALLABLE_MEMBER const T operator*(const vector3<T>& v1,
                                       const vector3<T>& v2) {
  return v1.dot(v2);
}

template <class T>
CUDA_CALLABLE_MEMBER const vector3<T> operator^(const vector3<T>& v1,
                                                const vector3<T>& v2) {
  return v1.cross(v2);
}

template <class T>
std::ostream& operator<<(std::ostream& os, const vector3<T> v) {
  os << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
  return os;
}

#endif  // VECTOR3_HPP_
