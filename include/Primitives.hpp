/**************************************************************************/ /**
 * \file Primitives.hpp
 * \brief Defines some macros, often used functions, and structures for
 *        coordinates and colors.
 * \todo Overload operators for each structure (maybe with a template?)
 * \author Nicolas Delbosc
 * \version 1.0a
 * \date December 2013
 *****************************************************************************/
#pragma once
#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#ifdef WITH_GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#endif
#include <stdlib.h>
#include <math.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
// #include <GL/glut.h>

#include "CudaUtils.hpp"

#define NaN std::numeric_limits<real>::quiet_NaN()

typedef int voxel;

template <class T>
inline void hash_combine(std::size_t &seed, const T &v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

/// The type real is to be used instead of float or double
//typedef double real;

/// Macro to print an error message.
#define PRINT_ERROR(texte)                                                                                                             \
  {                                                                                                                                    \
    std::cerr << "Error in " << __PRETTY_FUNCTION__ << " in file " << __FILE__ << ", line " << __LINE__ << ": " << texte << std::endl; \
  }
/// Macro to print an error message and terminate the program.
#define FATAL_ERROR(texte)                                                                                                                   \
  {                                                                                                                                          \
    std::cerr << "FATAL ERROR in " << __PRETTY_FUNCTION__ << " in file " << __FILE__ << ", line " << __LINE__ << ": " << texte << std::endl; \
    exit(-1);                                                                                                                                \
  }

/// Macro to print a warning message
/** If VERBOSE is undefined, it does not print anything. */
#if defined(VERBOSE)
#define PRINT_WARNING(texte)                                                                                                             \
  {                                                                                                                                      \
    std::cerr << "Warning in " << __PRETTY_FUNCTION__ << " in file " << __FILE__ << ", line " << __LINE__ << ": " << texte << std::endl; \
  }
#else
#define PRINT_WARNING(texte) \
  {                          \
  }
#endif

/// Macro to print a message for unfinished functions
#define PRINT_UNFINISHED()                                                                                                                      \
  {                                                                                                                                             \
    std::cerr << "Warning : " << __PRETTY_FUNCTION__ << " in file " << __FILE__ << ", line " << __LINE__ << ", is not finished. " << std::endl; \
  }

#ifdef WITH_OPENGL
// /// Macro to check if OpenGL has any error in its stack.
// /** Calls the function checkOpenGLerrors(). */
// #define CHECK_OPENGL_ERROR() (checkOpenGLerrors(__PRETTY_FUNCTION__, __FILE__, __LINE__))

// /// Function to check OpenGL error and print them.
// /** Do not use this function directly, use the macro CHECK_OPENGL_ERROR instead. */
// inline void checkOpenGLerrors(std::string function, std::string file, int line)
// {
//   GLenum error;
//   while ((error = glGetError()) != GL_NO_ERROR)
// #ifdef NO_GLU
//     std::cerr << "OpenGL error in " << function << " in file " << file << ", line " << line << ": " << error << std::endl;
// #else
//     std::cerr << "OpenGL error in " << function << " in file " << file << ", line " << line << ": "
//               << gluErrorString(error) << std::endl;
// #endif
// }

// #else
// #define CHECK_OPENGL_ERROR()
#endif //ifdef WITH_OPENGL

/// Compute the absolute value of a
template <class T>
inline T abs(const T &a) { return (a > 0) ? a : (-a); }
/// Compute the minimum of a and b
template <class T>
inline const T &min(const T &a, const T &b) { return (a < b) ? a : b; }
/// Compute the maximum of a and b
template <class T>
inline const T &max(const T &a, const T &b) { return (a > b) ? a : b; }

///// Empty structure, use it in place of a Buffer you don't want to use
//struct None {
//  static const bool empty = true;
//  static const int dimension = 0;
//};

template <typename T>
struct vec3;
/// Structure to regroup 2 numbers of type T. (useful for 2D coordinates)
template <typename T>
struct vec2
{
  static const bool empty = false;
  /// number of components
  static const int dimension = 2;
  /// define the null vector
  static const vec2 ZERO;
  /// define the base unit vector along x-axis
  static const vec2 X;
  /// define the base unit vector along y-axis
  static const vec2 Y;
  /// component of the vector along x-axis
  T x;
  /// component of the vector along y-axis
  T y;
  /// Default constructor
  vec2()
  {
    x = 0;
    y = 0;
  }
  /// Constructor
  vec2(T x, T y)
  {
    this->x = x;
    this->y = y;
  }
  //compute the norm
  inline T norm() const
  {
    return sqrt(x * x + y * y);
  }
  //normalise the vector (divide by its norm)
  inline void normalize()
  {
    T n = norm();
    if (n == 0)
    {
      PRINT_WARNING("Vector is zero")
      x = 1;
      y = 0;
    }
    else
    {
      x /= n;
      y /= n;
    }
  }
  //constructor with a vec3
  template <typename U>
  vec2<T>(vec3<U> v);
  /// Output a vector
  template <typename U>
  friend std::ostream &operator<<(std::ostream &out, const vec2<U> &v);
};

template <typename T>
const vec2<T> vec2<T>::ZERO = vec2<T>(0, 0);
template <typename T>
const vec2<T> vec2<T>::X = vec2<T>(1, 0);
template <typename T>
const vec2<T> vec2<T>::Y = vec2<T>(0, 1);

/// 2D vector of real
typedef vec2<real> vec2r;
/// 2D vector of unsigned int
typedef vec2<unsigned int> vec2ui;

/// Add two vectors together.
template <typename T>
inline vec2<T> operator+(const vec2<T> &v1, const vec2<T> &v2)
{
  return vec2<T>(v1.x + v2.x, v1.y + v2.y);
}
/// substract two vectors together.
template <typename T>
inline vec2<T> operator-(const vec2<T> &v1, const vec2<T> &v2)
{
  return vec2<T>(v1.x - v2.x, v1.y - v2.y);
}
/// Compute the scalar product of two vectors.
template <typename T>
inline T operator*(const vec2<T> &v1, const vec2<T> &v2)
{
  return v1.x * v2.x + v1.y * v2.y;
}
///compare 2 vectors
template <typename T>
inline bool operator!=(const vec2<T> &v1, const vec2<T> &v2)
{
  return ((v1.x != v2.x) || (v1.y != v2.y));
}
///compare 2 vectors
template <typename T>
inline bool operator==(const vec2<T> &v1, const vec2<T> &v2)
{
  return ((v1.x == v2.x) || (v1.y == v2.y));
}
/// Multiply a vector by a scalar
template <typename T>
inline vec2<T> operator*(double a, const vec2<T> &v)
{
  return vec2<T>(a * v.x, a * v.y);
}
/// Add a vector to the current vector
template <typename T>
inline void operator+=(vec2<T> &v1, const vec2<T> &v2)
{
  v1.x += v2.x;
  v1.y += v2.y;
}
/// Substract a vector to the current vector
template <typename T>
inline void operator-=(vec2<T> &v1, const vec2<T> &v2)
{
  v1.x -= v2.x;
  v1.y -= v2.y;
}

/// Output a vec2
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const vec2<T> &v)
{
  out << "(" << v.x << ", " << v.y << ")";
  return out;
}

struct pol3;

/// Structure to regroup 3 numbers of type T. (useful for 3D coordinates)
template <typename T>
struct vec3
{
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
  vec3()
  {
    x = 0;
    y = 0;
    z = 0;
  }
  /// Constructor
  vec3(T x, T y, T z)
  {
    this->x = x;
    this->y = y;
    this->z = z;
  }
  /// Constructor with another vec3
  template <typename U>
  vec3(vec3<U> v)
  {
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
  }

  //constructor with a pol3
  vec3(pol3 p);
  //constructor with a vec2
  template <typename U>
  vec3(vec2<U> v)
  {
    this->x = v.x;
    this->y = v.y;
    this->z = 0;
  }
  //compute the norm
  inline T norm() const
  {
    return sqrtf((*this) * (*this));
  }
  //normalise the vector (divide by its norm)
  inline void normalize()
  {
    T norm = sqrtf((*this) * (*this));
    if (norm == 0)
      PRINT_WARNING("Vector is zero")
    else
    {
      x /= norm;
      y /= norm;
      z /= norm;
    }
  }
#ifdef WITH_OPENGL
  // //call glTranslatef
  // inline void load() const { glTranslatef(x, y, z); }
#endif
#ifdef WITH_GLM
  //Transform to a glm::vec3
  inline glm::vec3 glm() const { return glm::vec3(x, y, z); }
  //Transform to a translation matrix glm::mat4
  inline glm::mat4 glmTranslate() const { return glm::translate(x, y, z); }
#endif
  /// Output a vector
  template <typename U>
  friend std::ostream &operator<<(std::ostream &out, const vec3<U> &v);
  /// Normalise a vector
  template <typename U>
  static inline vec3<U> normalize(const vec3<U> &v)
  {
    U norm = v.norm();
    if (norm == 0)
      PRINT_WARNING("Vector is zero")
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

//constructor of vec2 with a vec3
//TODO: fix it !
/*
template <typename T>
vec2<T>::vec2<T>(vec3<T> v)
{
  this->x = v.x;
  this->y = v.y;
}
*/

/// 3D vector of real
typedef vec3<real> vec3r;
/// 3D vector of unsigned int
typedef vec3<unsigned int> vec3ui;

/// reverse a vector
template <typename T>
inline vec3<T> operator-(const vec3<T> &v)
{
  return vec3<T>(-v.x, -v.y, -v.z);
}
/// Add two vectors together.
template <typename T>
inline vec3<T> operator+(const vec3<T> &v1, const vec3<T> &v2)
{
  return vec3<T>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
/// Add two vectors together.
template <typename T, typename U>
inline vec3r operator+(const vec3<T> &v1, const vec3<U> &v2)
{
  return vec3r(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
/// Add a vector to the current vector.
template <typename T>
inline void operator+=(vec3<T> &v1, const vec3<T> &v2)
{
  v1.x += v2.x;
  v1.y += v2.y;
  v1.z += v2.z;
}
/// substract two vectors together.
template <typename T>
inline vec3<T> operator-(const vec3<T> &v1, const vec3<T> &v2)
{
  return vec3<T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
/// Substract a vector to the current vector.
template <typename T>
inline void operator-=(vec3<T> &v1, const vec3<T> &v2)
{
  v1.x -= v2.x;
  v1.y -= v2.y;
  v1.z -= v2.z;
}
/// Compute the scalar product of two vectors.
template <typename T>
inline T operator*(const vec3<T> &v1, const vec3<T> &v2)
{
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
/// Compute the cross product of two vectors.
template <typename T>
inline vec3<T> operator^(const vec3<T> &v1, const vec3<T> &v2)
{
  return vec3<T>(v1.y * v2.z - v2.y * v1.z,
                 v1.z * v2.x - v2.z * v1.x,
                 v1.x * v2.y - v2.x * v1.y);
}
/// Multiply a vector by a scalar
template <typename T1, typename T2>
inline vec3<T2> operator*(const T1 &a, const vec3<T2> &v)
{
  return vec3<T2>(a * v.x, a * v.y, a * v.z);
}
/// Multiply a vector by a scalar
template <typename T1, typename T2>
inline void operator*=(vec3<T1> &v, const T2 &a)
{
  v.x *= a;
  v.y *= a;
  v.z *= a;
}

/// Divide a vector by a scalar
template <typename T1, typename T2>
inline vec3<T2> operator/(const vec3<T2> &v, const T1 &a)
{
  return vec3<T2>(v.x / a, v.y / a, v.z / a);
}
/// Divide a vector by a scalar
template <typename T1, typename T2>
inline void operator/=(vec3<T1> &v, const T2 &a)
{
  v.x /= a;
  v.y /= a;
  v.z /= a;
}

/// Output a vector
template <typename T>
std::ostream &operator<<(std::ostream &out, const vec3<T> &v)
{
  out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return out;
}

//let the compiler know of col4 structure (for col3 constructor)
//struct col4;

/// Structure to regroup 3 unsigned char. (useful for color components)
//template to avoid multiple definition and avoid a separate .cpp
//but T should only be unsigned char
template <typename T>
struct col3T
{
  static const bool empty = false;
  /// number of components
  static const int dimension = 3;
  /// define white color
  static const col3T white;
  /// define black color
  static const col3T black;
  /// define red color
  static const col3T red;
  /// define yellow color
  static const col3T yellow;
  /// define green color
  static const col3T green;
  /// define cyan color
  static const col3T cyan;
  /// define blue color
  static const col3T blue;
  /// define fuchsia color
  static const col3T fuchsia;
  /// red component
  T r;
  /// green component
  T g;
  /// blue component
  T b;
  /// Default constructor
  col3T() {}
  /// Constructor
  col3T(T r, T g, T b)
  {
    this->r = r;
    this->g = g;
    this->b = b;
  }
  /// Constructor with a col4
  //template < typename U >
  //col3T(col4<U> color);
  /// Constructor with an hexadecimal numberglColor3ub
  col3T(int hexValue)
  {
    r = ((hexValue >> 16) & 0xFF); // Extract the RR byte
    g = ((hexValue >> 8) & 0xFF);  // Extract the GG byte
    b = ((hexValue)&0xFF);         // Extract the BB byte
  }
#ifdef WITH_OPENGL
  // /// load the color in opengl
  // void load() const { glColor3ub(r, g, b); }
  // /// set the color as background (using glClearColor)
  // void loadAsBackground() const { glClearColor(r / 255., g / 255., b / 255., 0); }
#endif
  /// Output a color
  template <typename U>
  friend std::ostream &operator<<(std::ostream &out, const col3T<U> &c);
};

template <typename T>
const col3T<T> col3T<T>::black = col3T<T>(0, 0, 0);
template <typename T>
const col3T<T> col3T<T>::white = col3T<T>(255, 255, 255);
template <typename T>
const col3T<T> col3T<T>::red = col3T<T>(255, 0, 0);
template <typename T>
const col3T<T> col3T<T>::yellow = col3T<T>(255, 255, 0);
template <typename T>
const col3T<T> col3T<T>::green = col3T<T>(0, 255, 0);
template <typename T>
const col3T<T> col3T<T>::cyan = col3T<T>(0, 255, 255);
template <typename T>
const col3T<T> col3T<T>::blue = col3T<T>(0, 0, 255);
template <typename T>
const col3T<T> col3T<T>::fuchsia = col3T<T>(255, 0, 255);

//only specialisation of col3T that makes sense
typedef col3T<unsigned char> col3;

/// Multiply  a color by a scalar
/** Each component is cap to 255 */
template <typename T>
inline col3T<T> operator*(const real &number, const col3T<T> &color)
{
  return col3T<T>(min(real(255), number * color.r),
                  min(real(255), number * color.g),
                  min(real(255), number * color.b));
}
/// Add two colors
/** Each component is cap to 255 */
template <typename T>
inline col3T<T> operator+(const col3T<T> &c1, const col3T<T> &c2)
{
  return col3T<T>(min(255, c1.r + c2.r),
                  min(255, c1.g + c2.g),
                  min(255, c1.b + c2.b));
}
template <typename T>
inline void operator+=(col3T<T> &c1, const col3T<T> &c2)
{
  c1 = c1 + c2;
}
/// Blend two colors
/** Each component is cap to 255 */
template <typename T>
inline col3T<T> operator*(const col3T<T> &c1, const col3T<T> &c2)
{
  return col3T<T>(min(int(255), c1.r *c2.r),
                  min(int(255), c1.g *c2.g),
                  min(int(255), c1.b *c2.b));
}
/// Output a color
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const col3T<T> &c)
{
  out << "(" << int(c.r) << ", " << int(c.g) << ", " << int(c.b) << ")";
  return out;
}

/// Structure to regroup 4 unsigned char. (useful for color components)
template <typename T>
struct col4T
{
  static const bool empty = false;
  /// number of components
  static const int dimension = 4;
  /// define white color
  static const col4T white;
  /// define black color
  static const col4T black;
  /// define red color
  static const col4T red;
  /// define yellow color
  static const col4T yellow;
  /// define green color
  static const col4T green;
  /// define cyan color
  static const col4T cyan;
  /// define blue color
  static const col4T blue;
  /// define fuchsia color
  static const col4T fuchsia;
  /// red component
  T r;
  /// green component
  T g;
  /// blue component
  T b;
  /// alpha (transparency) compoment
  T a;
  /// Default constructor
  col4T() {}
  /// Constructor
  col4T(T r, T g, T b, T a)
  {
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
  }
  /// Constructor with a col3
  template <typename U>
  col4T(col3T<U> color)
  {
    this->r = color.r;
    this->g = color.g;
    this->b = color.b;
    this->a = 255;
  }
#ifdef WITH_OPENGL
  // /// load the color in opengl
  // void load() const { glColor4ub(r, g, b, a); }
  // /// set the color as background (using glClearColor)
  // void loadAsBackground() const { glClearColor(r / 255., g / 255., b / 255., a / 255.); }
#endif
  /// Output a color
  template <typename U>
  friend std::ostream &operator<<(std::ostream &out, const col4T<U> &c);
};

//Define some often used colors
template <typename T>
const col4T<T> col4T<T>::black = col4T<T>(0, 0, 0, 255);
template <typename T>
const col4T<T> col4T<T>::white = col4T<T>(255, 255, 255, 255);
template <typename T>
const col4T<T> col4T<T>::red = col4T<T>(255, 0, 0, 255);
template <typename T>
const col4T<T> col4T<T>::yellow = col4T<T>(255, 255, 0, 255);
template <typename T>
const col4T<T> col4T<T>::green = col4T<T>(0, 255, 0, 255);
template <typename T>
const col4T<T> col4T<T>::cyan = col4T<T>(0, 255, 255, 255);
template <typename T>
const col4T<T> col4T<T>::blue = col4T<T>(0, 0, 255, 255);
template <typename T>
const col4T<T> col4T<T>::fuchsia = col4T<T>(255, 0, 255, 255);

//only specialisation of col4T that makes sense
typedef col4T<unsigned char> col4;
/// Output a color
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const col4T<T> &c)
{
  out << "(" << int(c.r) << ", " << int(c.g) << ", " << int(c.b) << ", " << int(c.a) << ")";
  return out;
}

/// A structure to define a couple key-value.
/**
 * Allows to regroup any two class together,
 * useful to couple a key and its associated value.
 *
 * Example of use to define color keys:
 * \code
 *    typedef pair<real,col3> colorKey;
 * \endcode
 */
template <class T1, class T2>
struct pair_
{
  /// The key acts like an index
  T1 key;
  /// The value associated with the key
  T2 value;
  /// Default constructor
  pair_() {}
  /// Constructor
  pair_(T1 key, T2 value) : key(key), value(value)
  {
  }
};

/// Overload comparison operator on two pairs.
/** The comparison of two pairs is made by comparing their keys. */
template <class T1, class T2>
bool operator<(const pair_<T1, T2> &pair1, const pair_<T1, T2> &pair2)
{
  return (pair1.key < pair2.key);
}

/// Defines spherical coordinates (3D polar)
struct pol3
{
  ///radial distance always positive
  real r;
  ///polar angle in [0,PI]
  real theta;
  ///azimuth angle in [0,2*PI]
  real phi;
  /// Default constructor
  pol3() {}
  /// Constructor
  pol3(real r, real theta, real phi)
  {
    this->r = r;
    this->theta = theta;
    this->phi = phi;
  }
  /// Constructor with a vec3r
  pol3(const vec3r &v)
  {
    (*this) = v;
  }
  /// assigment operator with a vec3r
  pol3 &operator=(const vec3r &v)
  {
    this->r = sqrtf(v * v);
    if (this->r == 0)
    {
      this->theta = 0;
      this->phi = 0;
    }
    else
    {
      this->theta = acos(v.z / this->r);
      if (v.x == 0)
      {
        if (v.y > 0)
          this->phi = M_PI / 2;
        else
          this->phi = 3 * M_PI / 2;
      }
      else
      {
        this->phi = atan(v.y / v.x);
        if (v.x < 0)
          phi += M_PI;
      }
    }
    return *this;
  }
  /// set the radial distance
  /** if r is negative, it modfifies theta and phi accordingly*/
  inline void setR(real r)
  {
    if (r < 0)
    {
      this->r = -r;
      this->theta = M_PI - this->theta;
      this->phi += M_PI;
      if (phi > 2 * M_PI)
        this->phi -= 2 * M_PI;
    }
    else
      this->r = r;
  }
  /// set theta and cap it to [0,PI]
  inline void setTheta(real theta)
  {
    if (theta < 0)
      this->theta = 0;
    else if (theta > M_PI)
      this->theta = M_PI;
    else
      this->theta = theta;
  }
  /// set phi and cap it to [0,2*PI]
  inline void setPhi(real phi)
  {
    this->phi = phi;
    while (this->phi < 0)
      this->phi += 2 * M_PI;
    while (this->phi > 2 * M_PI)
      this->phi -= 2 * M_PI;
  }
  ///compute the corresponding cartesian coordinates
  inline vec3r getPos() const
  {
    return vec3r(r * sin(theta) * cos(phi),
                 r * sin(theta) * sin(phi),
                 r * cos(theta));
  }
  ///compute the "top of the head" direction for a camera
  inline vec3r getTop() const
  {
    real threshold = 0.0001;
    if (theta > M_PI - threshold)
      return vec3r(cos(phi), sin(phi), 0);
    else if (theta < threshold)
      return vec3r(-cos(phi), -sin(phi), 0);
    else
      return vec3r(0, 0, 1);
  }
#ifdef WITH_OPENGL
  // /// rotate using glRotatef
  // inline void load() const
  // {
  //   glRotatef(180. / M_PI * phi, 0, 0, 1);
  //   glRotatef(180. / M_PI * theta, 1, 0, 0);
  // }
#endif
  //normalise the vector (set the radius to 1)
  inline void normalize() { r = 1; }
};

/// substract two pol3 together.
/*
inline pol3 operator-( const pol3& p1, const pol3& p2)
{ 
  return pol3( 0.5*(p1.r+p2.r), p2.theta-p1.theta, p2.phi-p1.phi);
}
*/

template <typename T>
vec3<T>::vec3(pol3 p)
{
  this->x = p.r * sin(p.theta) * cos(p.phi);
  this->y = p.r * sin(p.theta) * sin(p.phi);
  this->z = p.r * cos(p.theta);
}

//couple of an origin and direction to define a ray
template <typename T>
struct ray
{
  T o; //origin
  T d; //direction
  /// Constructor
  ray(T origin, T direction) : o(origin), d(direction)
  {
  }
};

///structure to handle quadrilateron
struct quad
{
  vec3r p1, p2, p3, p4;
  /// Constructor
  quad(){};
  quad(vec3r point1, vec3r point2, vec3r point3, vec3r point4) : p1(point1), p2(point2), p3(point3), p4(point4)
  {
  }
  //check if the ray intersects the quad
  //compute the distance from the ray origin to the intersection point
  bool intersect(ray<vec3r> ray, real &distToIntersection) const
  {
    //find where the ray intersects the plane
    //base vectors
    vec3r x = (p2 - p1);
    x.normalize();
    vec3r y = (p4 - p1);
    y.normalize();
    //normal vector
    vec3r n = x ^ y;
    n.normalize();
    if ((ray.d * n) == 0) //ray is parallel to the plane
      return false;
    //intersection point
    distToIntersection = (p1 - ray.o) * n / (ray.d * n);
    vec3r p = ray.o + distToIntersection * ray.d;
    //project the intersection onto the quad base
    real xProj = (p - p1) * x;
    if ((xProj < 0) || (xProj > (p2 - p1).norm()))
      return false;
    real yProj = (p - p1) * y;
    if ((yProj < 0) || (yProj > (p4 - p1).norm()))
      return false;
    return true;
  }
#ifdef WITH_OPENGL
  // //display using OpenGL
  // void display() const
  // {
  //   glBegin(GL_QUADS);
  //   glVertex3f(p1.x, p1.y, p1.z);
  //   glVertex3f(p2.x, p2.y, p2.z);
  //   glVertex3f(p3.x, p3.y, p3.z);
  //   glVertex3f(p4.x, p4.y, p4.z);
  //   glEnd();
  // }
#endif
};

#endif
