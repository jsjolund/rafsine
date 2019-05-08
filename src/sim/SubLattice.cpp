#include "SubLattice.hpp"

D3Q4::Enum SubLattice::getDivisionAxis() const {
  return D3Q4::Y_AXIS;
  // int nx = getDims().x, ny = getDims().y, nz =
  // getDims().z; int xz = nx * nz, yz = ny * nz, xy = nx * ny; if (xy <=
  // xz && xy <= yz)
  //   return D3Q4::Z_AXIS;
  // else if (xz <= yz && xz <= xy)
  //   return D3Q4::Y_AXIS;
  // else
  //   return D3Q4::X_AXIS;
}

bool operator==(SubLattice const &a, SubLattice const &b) {
  return (a.getMin() == b.getMin() && a.getMax() == b.getMax() &&
          a.getHalo() == b.getHalo());
}

std::ostream &operator<<(std::ostream &os, const SubLattice p) {
  os << "size=" << p.getDims() << ", min=" << p.getMin()
     << ", max=" << p.getMax() << ", halo=" << p.getHalo();
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
  const D3Q4::Enum axis = oldSubLattices.at(0).getDivisionAxis();
  if (axis == D3Q4::X_AXIS) subLatticeCount->x *= factor;
  if (axis == D3Q4::Y_AXIS) subLatticeCount->y *= factor;
  if (axis == D3Q4::Z_AXIS) subLatticeCount->z *= factor;

  for (SubLattice subLattice : oldSubLattices) {
    glm::ivec3 min = subLattice.getMin(), max = subLattice.getMax(),
               halo = subLattice.getHalo();
    for (int i = 0; i < factor; i++) {
      float d = static_cast<float>(i + 1) / factor;
      switch (axis) {
        case D3Q4::X_AXIS:
          halo.x = haloSize;
          max.x = subLattice.getMin().x +
                  std::floor(1.0 * subLattice.getDims().x * d);
          break;
        case D3Q4::Y_AXIS:
          halo.y = haloSize;
          max.y = subLattice.getMin().y +
                  std::floor(1.0 * subLattice.getDims().y * d);
          break;
        case D3Q4::Z_AXIS:
          halo.z = haloSize;
          max.z = subLattice.getMin().z +
                  std::floor(1.0 * subLattice.getDims().z * d);
          break;
        default:
          break;
      }
      if (i == factor - 1) {
        max.x = subLattice.getMax().x;
        max.y = subLattice.getMax().y;
        max.z = subLattice.getMax().z;
      }
      subLattices->push_back(SubLattice(min, max, halo));
      switch (axis) {
        case D3Q4::X_AXIS:
          min.x = max.x;
          break;
        case D3Q4::Y_AXIS:
          min.y = max.y;
          break;
        case D3Q4::Z_AXIS:
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
                       unsigned int haloSize) const {
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
              if (a.getMin().z != b.getMin().z)
                return a.getMin().z < b.getMin().z;
              if (a.getMin().y != b.getMin().y)
                return a.getMin().y < b.getMin().y;
              return a.getMin().x < b.getMin().x;
            });
}

HaloSegment SubLattice::getHalo(glm::ivec3 direction,
                                SubLattice neighbour) const {
  HaloSegment halo;

  glm::ivec3 srcMin = glm::ivec3(0, 0, 0);
  glm::ivec3 srcMax = getArrayDims() - getHalo();
  glm::ivec3 dstMin = glm::ivec3(0, 0, 0);
  glm::ivec3 dstMax = neighbour.getArrayDims() - getHalo();
  glm::ivec3 srcDims = getArrayDims();
  glm::ivec3 dstDims = neighbour.getArrayDims();

  // Origin
  if (direction == glm::ivec3(0, 0, 0)) {
    halo.m_src = glm::ivec3(0, 0, 0);
    halo.m_dst = glm::ivec3(0, 0, 0);
    halo.m_srcStride = 0;
    halo.m_dstStride = 0;
    halo.m_segmentLength = 0;
    halo.m_numSegments = 0;
    return halo;

    // 6 faces
  } else if (direction == glm::ivec3(1, 0, 0)) {
    // YZ plane
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.y * srcDims.z;

  } else if (direction == glm::ivec3(-1, 0, 0)) {
    // YZ plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.y * srcDims.z;

  } else if (direction == glm::ivec3(0, 1, 0)) {
    // XZ plane
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = srcDims.x;
    halo.m_numSegments = srcDims.z;

  } else if (direction == glm::ivec3(0, -1, 0)) {
    // XZ plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = srcDims.x;
    halo.m_numSegments = srcDims.z;

  } else if (direction == glm::ivec3(0, 0, 1)) {
    // XY plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = srcDims.x * srcDims.y;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(0, 0, -1)) {
    // XY plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = srcDims.x * srcDims.y;
    halo.m_numSegments = 1;

    //////////////////////////////// 12 edges
  } else if (direction == glm::ivec3(1, 1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMax.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.z;

  } else if (direction == glm::ivec3(-1, -1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMax.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.z;

  } else if (direction == glm::ivec3(1, -1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.z;

  } else if (direction == glm::ivec3(-1, 1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x * srcDims.y;
    halo.m_dstStride = dstDims.x * dstDims.y;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.z;

  } else if (direction == glm::ivec3(1, 0, 1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.y;

  } else if (direction == glm::ivec3(-1, 0, -1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMax.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.y;

  } else if (direction == glm::ivec3(1, 0, -1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.y;

  } else if (direction == glm::ivec3(-1, 0, 1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = 1;
    halo.m_numSegments = srcDims.y;

  } else if (direction == glm::ivec3(0, 1, 1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = srcDims.x;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(0, -1, -1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMax.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = srcDims.x;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(0, 1, -1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = srcDims.x;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(0, -1, 1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_srcStride = srcDims.x;
    halo.m_dstStride = dstDims.x;
    halo.m_segmentLength = srcDims.x;
    halo.m_numSegments = 1;

    // 8 corners
  } else if (direction == glm::ivec3(1, 1, 1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMax.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(-1, -1, -1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMax.y, dstMax.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(-1, 1, 1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(1, -1, -1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMax.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(1, -1, 1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(-1, 1, -1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMax.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(1, 1, -1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else if (direction == glm::ivec3(-1, -1, 1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMax.y, dstMin.z);
    halo.m_srcStride = 1;
    halo.m_dstStride = 1;
    halo.m_segmentLength = 1;
    halo.m_numSegments = 1;

  } else {
    throw std::out_of_range("Unknown halo direction vector");
  }

  halo.m_src -= direction;
  halo.m_srcStride *= sizeof(real);
  halo.m_dstStride *= sizeof(real);
  halo.m_segmentLength *= sizeof(real);

  assert(halo.m_segmentLength <= halo.m_srcStride &&
         halo.m_segmentLength <= halo.m_dstStride);

  return halo;
}
