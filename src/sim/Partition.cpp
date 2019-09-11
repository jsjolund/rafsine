#include "Partition.hpp"

D3Q4::Enum Partition::getDivisionAxis() const {
  // glm::ivec3 n = getDims();
  // int xz = n.x * n.z;
  // int yz = n.y * n.z;
  // int xy = n.x * n.y;
  // if (xy <= xz && xy <= yz)
  //   return D3Q4::Z_AXIS;
  // else if (xz <= yz && xz <= xy)
  //   return D3Q4::Y_AXIS;
  // else
  //   return D3Q4::X_AXIS;
  return D3Q4::Y_AXIS;
}

int Partition::intersect(glm::ivec3 minIn, glm::ivec3 maxIn,
                          glm::ivec3 *minOut, glm::ivec3 *maxOut) const {
  minOut->x = max(minIn.x, m_min.x);
  minOut->y = max(minIn.y, m_min.y);
  minOut->z = max(minIn.z, m_min.z);
  maxOut->x = min(maxIn.x, m_max.x);
  maxOut->y = min(maxIn.y, m_max.y);
  maxOut->z = min(maxIn.z, m_max.z);
  glm::ivec3 d = *maxOut - *minOut;
  d.x = max(d.x, 0);
  d.y = max(d.y, 0);
  d.z = max(d.z, 0);
  return d.x * d.y * d.z;
}

bool operator==(Partition const &a, Partition const &b) {
  return (a.getMin() == b.getMin() && a.getMax() == b.getMax() &&
          a.getHalo() == b.getHalo());
}

std::ostream &operator<<(std::ostream &os, const Partition p) {
  os << "size=" << p.getDims() << ", min=" << p.getMin()
     << ", max=" << p.getMax() << ", halo=" << p.getHalo();
  return os;
}

std::ostream &operator<<(std::ostream &os, const HaloSegment p) {
  os << "src=" << p.m_src << ", dst=" << p.m_dst << ", spitch=" << p.m_spitch
     << ", dpitch=" << p.m_dpitch << ", width=" << p.m_width
     << ", height=" << p.m_height;
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

static void subdivide(int factor, glm::ivec3 *partitionCount,
                      std::vector<Partition> *partitions,
                      unsigned int haloSize) {
  std::vector<Partition> oldPartitions;
  oldPartitions.insert(oldPartitions.end(), partitions->begin(),
                        partitions->end());
  partitions->clear();
  const D3Q4::Enum axis = oldPartitions.at(0).getDivisionAxis();
  if (axis == D3Q4::X_AXIS) partitionCount->x *= factor;
  if (axis == D3Q4::Y_AXIS) partitionCount->y *= factor;
  if (axis == D3Q4::Z_AXIS) partitionCount->z *= factor;

  for (Partition partition : oldPartitions) {
    glm::ivec3 min = partition.getMin(), max = partition.getMax(),
               halo = partition.getHalo();
    for (int i = 0; i < factor; i++) {
      float d = static_cast<float>(i + 1) / factor;
      switch (axis) {
        case D3Q4::X_AXIS:
          halo.x = haloSize;
          max.x = partition.getMin().x +
                  std::floor(1.0 * partition.getDims().x * d);
          break;
        case D3Q4::Y_AXIS:
          halo.y = haloSize;
          max.y = partition.getMin().y +
                  std::floor(1.0 * partition.getDims().y * d);
          break;
        case D3Q4::Z_AXIS:
          halo.z = haloSize;
          max.z = partition.getMin().z +
                  std::floor(1.0 * partition.getDims().z * d);
          break;
        default:
          break;
      }
      if (i == factor - 1) {
        max.x = partition.getMax().x;
        max.y = partition.getMax().y;
        max.z = partition.getMax().z;
      }
      partitions->push_back(Partition(min, max, halo));
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

void Partition::split(unsigned int divisions, glm::ivec3 *partitionCount,
                       std::vector<Partition> *partitions,
                       unsigned int haloSize) const {
  partitions->clear();
  partitions->push_back(*this);

  if (divisions <= 1) return;
  std::vector<int> factors;
  primeFactors(divisions, &factors);
  std::reverse(factors.begin(), factors.end());
  for (int factor : factors)
    subdivide(factor, partitionCount, partitions, haloSize);

  std::sort(partitions->begin(), partitions->end(),
            [](Partition a, Partition b) {
              if (a.getMin().z != b.getMin().z)
                return a.getMin().z < b.getMin().z;
              if (a.getMin().y != b.getMin().y)
                return a.getMin().y < b.getMin().y;
              return a.getMin().x < b.getMin().x;
            });
}

HaloSegment Partition::getHalo(glm::ivec3 direction,
                                Partition neighbour) const {
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
    halo.m_spitch = 0;
    halo.m_dpitch = 0;
    halo.m_width = 0;
    halo.m_height = 0;
    return halo;

    // 6 faces
  } else if (direction == glm::ivec3(1, 0, 0)) {
    // YZ plane
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = 1;
    halo.m_height = srcDims.y * srcDims.z;

  } else if (direction == glm::ivec3(-1, 0, 0)) {
    // YZ plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = 1;
    halo.m_height = srcDims.y * srcDims.z;

  } else if (direction == glm::ivec3(0, 1, 0)) {
    // XZ plane
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = srcDims.x;
    halo.m_height = srcDims.z;

  } else if (direction == glm::ivec3(0, -1, 0)) {
    // XZ plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = srcDims.x;
    halo.m_height = srcDims.z;

  } else if (direction == glm::ivec3(0, 0, 1)) {
    // XY plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = srcDims.x * srcDims.y;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(0, 0, -1)) {
    // XY plane
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = srcDims.x * srcDims.y;
    halo.m_height = 1;

    //////////////////////////////// 12 edges
  } else if (direction == glm::ivec3(1, 1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMax.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = 1;
    halo.m_height = srcDims.z;

  } else if (direction == glm::ivec3(-1, -1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMax.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = 1;
    halo.m_height = srcDims.z;

  } else if (direction == glm::ivec3(1, -1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = 1;
    halo.m_height = srcDims.z;

  } else if (direction == glm::ivec3(-1, 1, 0)) {
    // Z edge
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x * srcDims.y;
    halo.m_dpitch = dstDims.x * dstDims.y;
    halo.m_width = 1;
    halo.m_height = srcDims.z;

  } else if (direction == glm::ivec3(1, 0, 1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = 1;
    halo.m_height = srcDims.y;

  } else if (direction == glm::ivec3(-1, 0, -1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMax.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = 1;
    halo.m_height = srcDims.y;

  } else if (direction == glm::ivec3(1, 0, -1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = 1;
    halo.m_height = srcDims.y;

  } else if (direction == glm::ivec3(-1, 0, 1)) {
    // Y edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = 1;
    halo.m_height = srcDims.y;

  } else if (direction == glm::ivec3(0, 1, 1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = srcDims.x;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(0, -1, -1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMax.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = srcDims.x;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(0, 1, -1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = srcDims.x;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(0, -1, 1)) {
    // X edge
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_spitch = srcDims.x;
    halo.m_dpitch = dstDims.x;
    halo.m_width = srcDims.x;
    halo.m_height = 1;

    // 8 corners
  } else if (direction == glm::ivec3(1, 1, 1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMax.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMin.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(-1, -1, -1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMax.y, dstMax.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(-1, 1, 1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMin.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(1, -1, -1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMax.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(1, -1, 1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMax.y, dstMin.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(-1, 1, -1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMin.y, dstMax.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(1, 1, -1)) {
    halo.m_src = glm::ivec3(srcMax.x, srcMax.y, srcMin.z);
    halo.m_dst = glm::ivec3(dstMin.x, dstMin.y, dstMax.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else if (direction == glm::ivec3(-1, -1, 1)) {
    halo.m_src = glm::ivec3(srcMin.x, srcMin.y, srcMax.z);
    halo.m_dst = glm::ivec3(dstMax.x, dstMax.y, dstMin.z);
    halo.m_spitch = 1;
    halo.m_dpitch = 1;
    halo.m_width = 1;
    halo.m_height = 1;

  } else {
    throw std::out_of_range("Unknown halo direction vector");
  }

  halo.m_src -= direction;
  halo.m_spitch *= sizeof(real);
  halo.m_dpitch *= sizeof(real);
  halo.m_width *= sizeof(real);

  assert(halo.m_width <= halo.m_spitch && halo.m_width <= halo.m_dpitch);

  return halo;
}
