#pragma once

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/vec3.hpp>

#include "CudaUtils.hpp"
#include "DdQq.hpp"
#include "Matrix3.hpp"
#include "Primitives.hpp"

namespace std {
template <>
struct hash<glm::ivec3> {
  std::size_t operator()(const glm::ivec3 &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.x);
    ::hash_combine(seed, p.y);
    ::hash_combine(seed, p.z);
    return seed;
  }
};
}  // namespace std

class SubLatticeSegment {
 public:
  glm::ivec3 m_src;
  glm::ivec3 m_dst;
  size_t m_srcStride;
  size_t m_dstStride;
  size_t m_segmentLength;
  size_t m_numSegments;

  SubLatticeSegment()
      : m_src(0, 0, 0),
        m_dst(0, 0, 0),
        m_srcStride(0),
        m_dstStride(0),
        m_segmentLength(0),
        m_numSegments(0) {}
};

class SubLattice {
 private:
  //! Minimum position in lattice
  glm::ivec3 m_min;
  //! Maximum position in lattice
  glm::ivec3 m_max;
  //! Size of halo buffers
  glm::ivec3 m_halo;
  //! Size of sub lattice
  glm::ivec3 m_size;
  //! Transformed size of sub lattice
  glm::ivec3 m_sizeT;
  //! Matrix dimension transform for calculating boundary elements
  Matrix3 m_transform;

 public:
  /**
   * @brief Calculate the coordinate of a boundary element on the faces of a 3D
   * matrix from an index between 0 and the sum of the numbers of elements on
   * each face. Matrix dimensions must be nx >= ny >= nz. Note that corners and
   * edges are repeated.
   *
   * @param i Index in range [0, 2 * (nx * ny + nx * nz + ny * nz) )
   * @param nx Matrix length on X-axis, range [1, ?]
   * @param ny Matrix length on Y-axis, range [1, nx]
   * @param nz Matrix length on Z-axis, range [1, ny]
   * @param x Matrix boundary face coordinate on X-axis
   * @param y Matrix boundary face coordinate on Y-axis
   * @param z Matrix boundary face coordinate on Z-axis
   */
  CUDA_CALLABLE_MEMBER static void getBoundaryElement(const int i, const int nx,
                                                      const int ny,
                                                      const int nz, int *x,
                                                      int *y, int *z) {
    const int n = nx * ny + nx * nz + ny * nz;
    const int j = i % n;
    const int xy = (2 * nx * ny) / (j + nx * ny + 1);
    const int xy_xz = (2 * nx * ny + 2 * nx * nz) / (j + nx * ny + nx * nz + 1);
    const int xy_xz_yz = (2 * n) / (j + n + 1);
    const int xz = xy_xz - xy;
    const int yz = xy_xz_yz - xy_xz;
    const int xy_x = j % nx;
    const int xy_y = j / nx;
    const int xy_z = (nz - 1) * (i / n);
    const int xz_x = j % nx;
    const int xz_y = (ny - 1) * (i / n);
    const int xz_z = (j % (nx * ny)) / nx;
    const int yz_x = (nx - 1) * (i / n);
    const int yz_y = (j % (nx * ny + nx * nz)) % ny;
    const int yz_z = (j % (nx * ny + nx * nz)) / ny;
    *x = xy * xy_x + xz * xz_x + yz * yz_x;
    *y = xy * xy_y + xz * xz_y + yz * yz_y;
    *z = xy * xy_z + xz * xz_z + yz * yz_z;
  }

  /**
   * @brief Read/write 1D access to matrix boundary face elements
   *
   * @param i Index in range [0, 2 * (nx * ny + nx * nz + ny * nz) )
   * @return T& The element
   */
  CUDA_CALLABLE_MEMBER void getBoundaryElement(int i, int *x, int *y,
                                               int *z) const {
    getBoundaryElement(i, m_sizeT.x, m_sizeT.y, m_sizeT.z, x, y, z);
    m_transform.mulVec(x, y, z);
  }

  /**
   * @brief Construct a new SubLattice object
   *
   * @param min Minimum point in lattice
   * @param max Maximum point in lattice
   */
  inline SubLattice(glm::ivec3 min, glm::ivec3 max, glm::ivec3 halo)
      : m_min(min),
        m_max(max),
        m_halo(halo),
        m_size(max - min),
        m_sizeT(max - min),
        m_transform() {
    // Calculate a transform such that nx >= ny >= nz
    if (m_size.x > m_size.y) {
      if (m_size.y > m_size.z) {
        return;  // Nothing to do
      } else if (m_size.x > m_size.z) {
        m_transform.swapRows(1, 2);
      } else {
        m_transform.swapRows(0, 2);
        m_transform.swapRows(1, 2);
      }
    } else {
      if (m_size.x > m_size.z) {
        m_transform.swapRows(0, 1);
      } else if (m_size.z > m_size.y) {
        m_transform.swapRows(0, 2);
      } else {
        m_transform.swapRows(0, 1);
        m_transform.swapRows(1, 2);
      }
    }
    // Apply the transform to matrix dimensions
    m_transform.mulVec(&m_sizeT.x, &m_sizeT.y, &m_sizeT.z);
    // Set the transform to reverse the mapping
    m_transform.transpose();
  }
  /**
   * @brief Construct a new empty SubLattice
   *
   */
  inline SubLattice()
      : m_min(0, 0, 0),
        m_max(0, 0, 0),
        m_halo(0, 0, 0),
        m_size(0, 0, 0),
        m_sizeT(0, 0, 0),
        m_transform() {}
  /**
   * @brief Copy constructor
   * @param other Another subLattice
   */
  inline SubLattice(const SubLattice &other)
      : m_min(other.m_min),
        m_max(other.m_max),
        m_halo(other.m_halo),
        m_size(other.m_size),
        m_sizeT(other.m_sizeT),
        m_transform(other.m_transform) {}

  /**
   * @brief Check if volume of sublattice is zero
   *
   * @return true
   * @return false
   */
  inline bool isEmpty() const {
    return m_min == glm::ivec3(0, 0, 0) && m_max == glm::ivec3(0, 0, 0) &&
           m_halo == glm::ivec3(0, 0, 0);
  }
  /**
   * @brief Get the minimum point of subLattice on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getMin() const { return m_min; }
  /**
   * @brief Get the maximum point of subLattice on the lattice
   *
   * @return glm::ivec3
   */
  inline glm::ivec3 getMax() const { return m_max; }
  /**
   * @brief Get the size of the halo in three dimensions
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getHalo() const { return m_halo; }
  /**
   * @brief Get the 3D sizes of the subLattice on the lattice
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getDims() const { return m_size; }
  /**
   * @brief Get the total size of the subLattice on the lattice
   *
   * @return size_t
   */
  CUDA_CALLABLE_MEMBER inline size_t getSize() const {
    glm::ivec3 dims = getDims();
    return dims.x * dims.y * dims.z;
  }
  CUDA_CALLABLE_MEMBER inline size_t getNumBoundaryElements() const {
    return 2 *
           (m_size.x * m_size.y + m_size.x * m_size.z + m_size.y * m_size.z);
  }
  /**
   * @brief Get the 3D array dimensions of the first order q of the distribution
   * function (including halos)
   *
   * @return glm::ivec3
   */
  CUDA_CALLABLE_MEMBER inline glm::ivec3 getArrayDims() const {
    glm::ivec3 dims = getDims();
    return dims + m_halo * 2;
  }
  /**
   * @brief Get the array size of the first order q of the distribution
   * function (including halos), or in other words, the array stride between
   * different q > 1
   *
   * @return glm::ivec3
   */
  inline size_t getArrayStride() const {
    glm::ivec3 dims = getArrayDims();
    return dims.x * dims.y * dims.z;
  }

  /**
   * @brief Finds the axis with the least slice area when cut
   *
   * @return SubLattice::Enum The axis
   */
  D3Q7::Enum getDivisionAxis() const;

  void getHaloPlane(glm::ivec3 direction, glm::ivec3 *src, size_t *srcStride,
                    glm::ivec3 srcDim, glm::ivec3 *dst, size_t *dstStride,
                    glm::ivec3 dstDim, size_t *width, size_t *height) const;

  SubLatticeSegment getSubLatticeSegment(glm::ivec3 direction,
                                         SubLattice neighbour) const;

  void split(unsigned int divisions, glm::ivec3 *subLatticeCount,
             std::vector<SubLattice> *subLattices, unsigned int haloSize) const;
};
bool operator==(SubLattice const &a, SubLattice const &b);
std::ostream &operator<<(std::ostream &os, SubLattice p);

namespace std {
template <>
struct hash<SubLattice> {
  std::size_t operator()(const SubLattice &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, p.getMin().x);
    ::hash_combine(seed, p.getMin().y);
    ::hash_combine(seed, p.getMin().z);
    ::hash_combine(seed, p.getMax().x);
    ::hash_combine(seed, p.getMax().y);
    ::hash_combine(seed, p.getMax().z);
    ::hash_combine(seed, p.getHalo().x);
    ::hash_combine(seed, p.getHalo().y);
    ::hash_combine(seed, p.getHalo().z);
    return seed;
  }
};
}  // namespace std
