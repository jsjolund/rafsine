#include "Partition.hpp"

unsigned int Partition::intersect(Vector3<unsigned int> minIn,
                                  Vector3<unsigned int> maxIn,
                                  Vector3<unsigned int>* minOut,
                                  Vector3<unsigned int>* maxOut) const {
  minOut->x() = max(minIn.x(), m_min.x());
  minOut->y() = max(minIn.y(), m_min.y());
  minOut->z() = max(minIn.z(), m_min.z());
  maxOut->x() = min(maxIn.x(), m_max.x());
  maxOut->y() = min(maxIn.y(), m_max.y());
  maxOut->z() = min(maxIn.z(), m_max.z());
  Vector3<unsigned int> d = *maxOut - *minOut;
  d.x() = max(d.x(), 0);
  d.y() = max(d.y(), 0);
  d.z() = max(d.z(), 0);
  return d.x() * d.y() * d.z();
}

bool operator==(Partition const& a, Partition const& b) {
  return (a.getMin() == b.getMin() && a.getMax() == b.getMax() &&
          a.getGhostLayer() == b.getGhostLayer());
}

std::ostream& operator<<(std::ostream& os, const Partition p) {
  os << "size=" << p.getExtents() << ", min=" << p.getMin()
     << ", max=" << p.getMax() << ", ghostLayer=" << p.getGhostLayer();
  return os;
}

std::ostream& operator<<(std::ostream& os, const GhostLayerParameters p) {
  os << "src=" << p.m_src << ", dst=" << p.m_dst << ", spitch=" << p.m_spitch
     << ", dpitch=" << p.m_dpitch << ", width=" << p.m_width
     << ", height=" << p.m_height;
  return os;
}

// static void primeFactors(int n, std::vector<int>* factors) {
//   while (n % 2 == 0) {
//     factors->push_back(2);
//     n = n / 2;
//   }
//   for (int i = 3; i <= sqrt(n); i = i + 2) {
//     while (n % i == 0) {
//       factors->push_back(i);
//       n = n / i;
//     }
//   }
//   if (n > 2) factors->push_back(n);
// }

static void subdivide(size_t factor,
                      Vector3<int>* partitionCount,
                      std::vector<Partition>* partitions,
                      unsigned int ghostLayerSize,
                      D3Q4::Enum axis) {
  std::vector<Partition> oldPartitions;
  oldPartitions.insert(oldPartitions.end(), partitions->begin(),
                       partitions->end());
  partitions->clear();
  if (axis == D3Q4::X_AXIS) partitionCount->x() *= factor;
  if (axis == D3Q4::Y_AXIS) partitionCount->y() *= factor;
  if (axis == D3Q4::Z_AXIS) partitionCount->z() *= factor;

  for (Partition partition : oldPartitions) {
    Vector3<unsigned int> min = partition.getMin(), max = partition.getMax();
    Vector3<size_t> ghostLayer = partition.getGhostLayer();
    for (size_t i = 0; i < factor; i++) {
      float d = static_cast<float>(i + 1) / factor;
      switch (axis) {
        case D3Q4::X_AXIS:
          ghostLayer.x() = ghostLayerSize;
          max.x() = partition.getMin().x() +
                    std::floor(1.0 * partition.getExtents().x() * d);
          break;
        case D3Q4::Y_AXIS:
          ghostLayer.y() = ghostLayerSize;
          max.y() = partition.getMin().y() +
                    std::floor(1.0 * partition.getExtents().y() * d);
          break;
        case D3Q4::Z_AXIS:
          ghostLayer.z() = ghostLayerSize;
          max.z() = partition.getMin().z() +
                    std::floor(1.0 * partition.getExtents().z() * d);
          break;
        default:
          break;
      }
      if (i == factor - 1) {
        max.x() = partition.getMax().x();
        max.y() = partition.getMax().y();
        max.z() = partition.getMax().z();
      }
      partitions->push_back(Partition(min, max, ghostLayer));
      switch (axis) {
        case D3Q4::X_AXIS:
          min.x() = max.x();
          break;
        case D3Q4::Y_AXIS:
          min.y() = max.y();
          break;
        case D3Q4::Z_AXIS:
          min.z() = max.z();
          break;
        default:
          break;
      }
    }
  }
}

void Partition::split(std::vector<Partition>* partitions,
                      Vector3<int>* partitionCount,
                      unsigned int nd,
                      unsigned int ghostLayerSize,
                      D3Q4::Enum partitioning) const {
  partitions->clear();
  partitions->push_back(*this);
  if (nd <= 1) return;
  subdivide(nd, partitionCount, partitions, ghostLayerSize, partitioning);
  std::sort(partitions->begin(), partitions->end(),
            [](Partition a, Partition b) {
              if (a.getMin().z() != b.getMin().z())
                return a.getMin().z() < b.getMin().z();
              if (a.getMin().y() != b.getMin().y())
                return a.getMin().y() < b.getMin().y();
              return a.getMin().x() < b.getMin().x();
            });
}

GhostLayerParameters Partition::getGhostLayer(Vector3<int> direction,
                                              Partition neighbour) const {
  GhostLayerParameters ghostLayer;

  Vector3<size_t> srcMin(0, 0, 0);
  Vector3<size_t> srcMax = getArrayExtents() - getGhostLayer();
  Vector3<size_t> dstMin(0, 0, 0);
  Vector3<size_t> dstMax = neighbour.getArrayExtents() - getGhostLayer();
  Vector3<size_t> srcExtents = getArrayExtents();
  Vector3<size_t> dstExtents = neighbour.getArrayExtents();

  // Origin
  if (direction == Vector3<int>(0, 0, 0)) {
    ghostLayer.m_src = Vector3<size_t>(0, 0, 0);
    ghostLayer.m_dst = Vector3<size_t>(0, 0, 0);
    ghostLayer.m_spitch = 0;
    ghostLayer.m_dpitch = 0;
    ghostLayer.m_width = 0;
    ghostLayer.m_height = 0;
    return ghostLayer;

    // 6 faces
  } else if (direction == Vector3<int>(1, 0, 0)) {
    // YZ plane
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.y() * srcExtents.z();

  } else if (direction == Vector3<int>(-1, 0, 0)) {
    // YZ plane
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.y() * srcExtents.z();

  } else if (direction == Vector3<int>(0, 1, 0)) {
    // XZ plane
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMax.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = srcExtents.x();
    ghostLayer.m_height = srcExtents.z();

  } else if (direction == Vector3<int>(0, -1, 0)) {
    // XZ plane
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMax.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = srcExtents.x();
    ghostLayer.m_height = srcExtents.z();

  } else if (direction == Vector3<int>(0, 0, 1)) {
    // XY plane
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = srcExtents.x() * srcExtents.y();
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(0, 0, -1)) {
    // XY plane
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMax.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = srcExtents.x() * srcExtents.y();
    ghostLayer.m_height = 1;

    //////////////////////////////// 12 edges
  } else if (direction == Vector3<int>(1, 1, 0)) {
    // Z edge
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMax.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.z();

  } else if (direction == Vector3<int>(-1, -1, 0)) {
    // Z edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMax.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.z();

  } else if (direction == Vector3<int>(1, -1, 0)) {
    // Z edge
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMax.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.z();

  } else if (direction == Vector3<int>(-1, 1, 0)) {
    // Z edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMax.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x() * srcExtents.y();
    ghostLayer.m_dpitch = dstExtents.x() * dstExtents.y();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.z();

  } else if (direction == Vector3<int>(1, 0, 1)) {
    // Y edge
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMin.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.y();

  } else if (direction == Vector3<int>(-1, 0, -1)) {
    // Y edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMin.y(), dstMax.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.y();

  } else if (direction == Vector3<int>(1, 0, -1)) {
    // Y edge
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMax.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.y();

  } else if (direction == Vector3<int>(-1, 0, 1)) {
    // Y edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = 1;
    ghostLayer.m_height = srcExtents.y();

  } else if (direction == Vector3<int>(0, 1, 1)) {
    // X edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMax.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = srcExtents.x();
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(0, -1, -1)) {
    // X edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMax.y(), dstMax.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = srcExtents.x();
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(0, 1, -1)) {
    // X edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMax.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMax.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = srcExtents.x();
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(0, -1, 1)) {
    // X edge
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMax.y(), dstMin.z());
    ghostLayer.m_spitch = srcExtents.x();
    ghostLayer.m_dpitch = dstExtents.x();
    ghostLayer.m_width = srcExtents.x();
    ghostLayer.m_height = 1;

    // 8 corners
  } else if (direction == Vector3<int>(1, 1, 1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMax.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(-1, -1, -1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMax.y(), dstMax.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(-1, 1, 1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMax.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMin.y(), dstMin.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(1, -1, -1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMin.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMax.y(), dstMax.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(1, -1, 1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMin.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMax.y(), dstMin.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(-1, 1, -1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMax.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMin.y(), dstMax.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(1, 1, -1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMax.x(), srcMax.y(), srcMin.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMin.x(), dstMin.y(), dstMax.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else if (direction == Vector3<int>(-1, -1, 1)) {
    ghostLayer.m_src = Vector3<size_t>(srcMin.x(), srcMin.y(), srcMax.z());
    ghostLayer.m_dst = Vector3<size_t>(dstMax.x(), dstMax.y(), dstMin.z());
    ghostLayer.m_spitch = 1;
    ghostLayer.m_dpitch = 1;
    ghostLayer.m_width = 1;
    ghostLayer.m_height = 1;

  } else {
    throw std::out_of_range("Unknown ghostLayer direction vector");
  }

  ghostLayer.m_src.x() -= direction.x();
  ghostLayer.m_src.y() -= direction.y();
  ghostLayer.m_src.z() -= direction.z();
  ghostLayer.m_spitch *= sizeof(real_t);
  ghostLayer.m_dpitch *= sizeof(real_t);
  ghostLayer.m_width *= sizeof(real_t);

  assert(ghostLayer.m_width <= ghostLayer.m_spitch &&
         ghostLayer.m_width <= ghostLayer.m_dpitch);

  return ghostLayer;
}
