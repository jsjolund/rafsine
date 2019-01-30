#include "SubLattice.hpp"

D3Q7::Enum SubLattice::getDivisionAxis() {
  return D3Q7::Y_AXIS_POS;
  // int nx = getLatticeDims().x, ny = getLatticeDims().y, nz =
  // getLatticeDims().z; int xz = nx * nz, yz = ny * nz, xy = nx * ny; if (xy <=
  // xz && xy <= yz)
  //   return D3Q7::Z_AXIS_POS;
  // else if (xz <= yz && xz <= xy)
  //   return D3Q7::Y_AXIS_POS;
  // else
  //   return D3Q7::X_AXIS_POS;
}

bool operator==(SubLattice const &a, SubLattice const &b) {
  return (a.getLatticeMin() == b.getLatticeMin() &&
          a.getLatticeMax() == b.getLatticeMax());
}

std::ostream &operator<<(std::ostream &os, const SubLattice p) {
  os << "min=" << p.getLatticeMin() << ", max=" << p.getLatticeMax()
     << ", halo=" << p.getHalo();
  return os;
}

static void primeFactors(int n, std::vector<int> *factors) {
  while (n % 2 == 0) {
    factors->push_back(2);
    n = n / 2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    while (n % i == 0) {
      factors->push_back(i);
      n = n / i;
    }
  }
  if (n > 2) factors->push_back(n);
}

static void subdivide(int factor, glm::ivec3 *subLatticeCount,
                      std::vector<SubLattice> *subLattices,
                      unsigned int haloSize) {
  std::vector<SubLattice> oldSubLattices;
  oldSubLattices.insert(oldSubLattices.end(), subLattices->begin(),
                        subLattices->end());
  subLattices->clear();
  const D3Q7::Enum axis = oldSubLattices.at(0).getDivisionAxis();
  if (axis == D3Q7::X_AXIS_POS) subLatticeCount->x *= factor;
  if (axis == D3Q7::Y_AXIS_POS) subLatticeCount->y *= factor;
  if (axis == D3Q7::Z_AXIS_POS) subLatticeCount->z *= factor;

  for (SubLattice subLattice : oldSubLattices) {
    glm::ivec3 min = subLattice.getLatticeMin(),
               max = subLattice.getLatticeMax(), halo = subLattice.getHalo();
    for (int i = 0; i < factor; i++) {
      float d = static_cast<float>(i + 1) / factor;
      switch (axis) {
        case D3Q7::X_AXIS_POS:
          halo.x = haloSize;
          max.x = subLattice.getLatticeMin().x +
                  std::ceil(1.0 * subLattice.getLatticeDims().x * d);
          break;
        case D3Q7::Y_AXIS_POS:
          halo.y = haloSize;
          max.y = subLattice.getLatticeMin().y +
                  std::ceil(1.0 * subLattice.getLatticeDims().y * d);
          break;
        case D3Q7::Z_AXIS_POS:
          halo.z = haloSize;
          max.z = subLattice.getLatticeMin().z +
                  std::ceil(1.0 * subLattice.getLatticeDims().z * d);
          break;
        default:
          break;
      }
      if (i == factor - 1) {
        max.x = subLattice.getLatticeMax().x;
        max.y = subLattice.getLatticeMax().y;
        max.z = subLattice.getLatticeMax().z;
      }
      subLattices->push_back(SubLattice(min, max, halo));
      switch (axis) {
        case D3Q7::X_AXIS_POS:
          min.x = max.x;
          break;
        case D3Q7::Y_AXIS_POS:
          min.y = max.y;
          break;
        case D3Q7::Z_AXIS_POS:
          min.z = max.z;
          break;
        default:
          break;
      }
    }
  }
}

void SubLattice::split(unsigned int divisions, glm::ivec3 *subLatticeCount,
                       std::vector<SubLattice> *subLattices,
                       unsigned int haloSize) {
  subLattices->clear();
  subLattices->push_back(*this);

  if (divisions <= 1) return;
  std::vector<int> factors;
  primeFactors(divisions, &factors);
  std::reverse(factors.begin(), factors.end());
  for (int factor : factors)
    subdivide(factor, subLatticeCount, subLattices, haloSize);

  std::sort(subLattices->begin(), subLattices->end(),
            [](SubLattice a, SubLattice b) {
              if (a.getLatticeMin().z != b.getLatticeMin().z)
                return a.getLatticeMin().z < b.getLatticeMin().z;
              if (a.getLatticeMin().y != b.getLatticeMin().y)
                return a.getLatticeMin().y < b.getLatticeMin().y;
              return a.getLatticeMin().x < b.getLatticeMin().x;
            });
}

int SubLattice::toLocalIndex(unsigned int df_idx, int x, int y, int z) {
  glm::ivec3 p(x, y, z);
  glm::ivec3 min = getLatticeMin() - getHalo();
  glm::ivec3 max = getLatticeMax() + getHalo();
  if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
      p.y < max.y && p.z < max.z) {
    p = p - getLatticeMin() + getHalo();
    glm::ivec3 n = getLatticeDims() + getHalo() * 2;
    int idx = I4D(df_idx, p.x, p.y, p.z, n.x, n.y, n.z);
    return idx;
  }
  throw std::out_of_range("Invalid range");
}

SubLatticeSegment SubLattice::getSubLatticeSegment(glm::ivec3 direction,
                                                   SubLattice neighbour) {
  SubLatticeSegment segment;
  if (direction == glm::ivec3(0, 0, 0)) return segment;

  getHaloPlane(direction, &segment.m_src, &segment.m_srcStride, getArrayDims(),
               &segment.m_dst, &segment.m_dstStride, neighbour.getArrayDims(),
               &segment.m_segmentLength, &segment.m_numSegments);

  segment.m_src -= direction;
  segment.m_srcStride *= sizeof(real);
  segment.m_dstStride *= sizeof(real);
  segment.m_segmentLength *= sizeof(real);

  assert(segment.m_segmentLength <= segment.m_srcStride &&
         segment.m_segmentLength <= segment.m_dstStride);

  return segment;
}

void SubLattice::getHaloPlane(glm::ivec3 direction, glm::ivec3 *src,
                              size_t *srcStride, glm::ivec3 srcDim,
                              glm::ivec3 *dst, size_t *dstStride,
                              glm::ivec3 dstDim, size_t *width,
                              size_t *height) {
  glm::ivec3 amin = glm::ivec3(0, 0, 0);
  glm::ivec3 amax = srcDim - getHalo();
  glm::ivec3 bmin = glm::ivec3(0, 0, 0);
  glm::ivec3 bmax = dstDim - getHalo();
  glm::ivec3 a = srcDim;
  glm::ivec3 b = dstDim;

  // Origin
  if (direction == glm::ivec3(0, 0, 0)) {
    *src = glm::ivec3(0, 0, 0);
    *dst = glm::ivec3(0, 0, 0);
    *srcStride = 0;
    *dstStride = 0;
    *width = 0;
    *height = 0;
    return;

    // 6 faces
  } else if (direction == glm::ivec3(1, 0, 0)) {
    // YZ plane
    *src = glm::ivec3(amax.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = 1;
    *height = a.y * a.z;

  } else if (direction == glm::ivec3(-1, 0, 0)) {
    // YZ plane
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmax.x, bmin.y, bmin.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = 1;
    *height = a.y * a.z;

  } else if (direction == glm::ivec3(0, 1, 0)) {
    // XZ plane
    *src = glm::ivec3(amin.x, amax.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = a.x;
    *height = a.z;

  } else if (direction == glm::ivec3(0, -1, 0)) {
    // XZ plane
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmax.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = a.x;
    *height = a.z;

  } else if (direction == glm::ivec3(0, 0, 1)) {
    // XY plane
    *src = glm::ivec3(amin.x, amin.y, amax.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = a.x * a.y;
    *height = 1;

  } else if (direction == glm::ivec3(0, 0, -1)) {
    // XY plane
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmax.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = a.x * a.y;
    *height = 1;

    //////////////////////////////// 12 edges
  } else if (direction == glm::ivec3(1, 1, 0)) {
    // Z edge
    *src = glm::ivec3(amax.x, amax.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = 1;
    *height = a.z;

  } else if (direction == glm::ivec3(-1, -1, 0)) {
    // Z edge
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmax.x, bmax.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = 1;
    *height = a.z;

  } else if (direction == glm::ivec3(1, -1, 0)) {
    // Z edge
    *src = glm::ivec3(amax.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmax.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = 1;
    *height = a.z;

  } else if (direction == glm::ivec3(-1, 1, 0)) {
    // Z edge
    *src = glm::ivec3(amin.x, amax.y, amin.z);
    *dst = glm::ivec3(bmax.x, bmin.y, bmin.z);
    *srcStride = a.x * a.y;
    *dstStride = b.x * b.y;
    *width = 1;
    *height = a.z;

  } else if (direction == glm::ivec3(1, 0, 1)) {
    // Y edge
    *src = glm::ivec3(amax.x, amin.y, amax.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = 1;
    *height = a.y;

  } else if (direction == glm::ivec3(-1, 0, -1)) {
    // Y edge
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmax.x, bmin.y, bmax.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = 1;
    *height = a.y;

  } else if (direction == glm::ivec3(1, 0, -1)) {
    // Y edge
    *src = glm::ivec3(amax.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmax.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = 1;
    *height = a.y;

  } else if (direction == glm::ivec3(-1, 0, 1)) {
    // Y edge
    *src = glm::ivec3(amin.x, amin.y, amax.z);
    *dst = glm::ivec3(bmax.x, bmin.y, bmin.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = 1;
    *height = a.y;

  } else if (direction == glm::ivec3(0, 1, 1)) {
    // X edge
    *src = glm::ivec3(amin.x, amax.y, amax.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = a.x;
    *height = 1;

  } else if (direction == glm::ivec3(0, -1, -1)) {
    // X edge
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmax.y, bmax.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = a.x;
    *height = 1;

  } else if (direction == glm::ivec3(0, 1, -1)) {
    // X edge
    *src = glm::ivec3(amin.x, amax.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmax.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = a.x;
    *height = 1;

  } else if (direction == glm::ivec3(0, -1, 1)) {
    // X edge
    *src = glm::ivec3(amin.x, amin.y, amax.z);
    *dst = glm::ivec3(bmin.x, bmax.y, bmin.z);
    *srcStride = a.x;
    *dstStride = b.x;
    *width = a.x;
    *height = 1;

    // 8 corners
  } else if (direction == glm::ivec3(1, 1, 1)) {
    *src = glm::ivec3(amax.x, amax.y, amax.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmin.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(-1, -1, -1)) {
    *src = glm::ivec3(amin.x, amin.y, amin.z);
    *dst = glm::ivec3(bmax.x, bmax.y, bmax.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(-1, 1, 1)) {
    *src = glm::ivec3(amin.x, amax.y, amax.z);
    *dst = glm::ivec3(bmax.x, bmin.y, bmin.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(1, -1, -1)) {
    *src = glm::ivec3(amax.x, amin.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmax.y, bmax.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(1, -1, 1)) {
    *src = glm::ivec3(amax.x, amin.y, amax.z);
    *dst = glm::ivec3(bmin.x, bmax.y, bmin.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(-1, 1, -1)) {
    *src = glm::ivec3(amin.x, amax.y, amin.z);
    *dst = glm::ivec3(bmax.x, bmin.y, bmax.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(1, 1, -1)) {
    *src = glm::ivec3(amax.x, amax.y, amin.z);
    *dst = glm::ivec3(bmin.x, bmin.y, bmax.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else if (direction == glm::ivec3(-1, -1, 1)) {
    *src = glm::ivec3(amin.x, amin.y, amax.z);
    *dst = glm::ivec3(bmax.x, bmax.y, bmin.z);
    *srcStride = 1;
    *dstStride = 1;
    *width = 1;
    *height = 1;

  } else {
    throw std::out_of_range("Unknown halo direction vector");
  }
}
