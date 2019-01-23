#include "DistributionFunction.hpp"

DistributionFunction::DistributionFunction(unsigned int Q,
                                           unsigned int latticeSizeX,
                                           unsigned int latticeSizeY,
                                           unsigned int latticeSizeZ,
                                           unsigned int subdivisions)
    : Lattice(Q, latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions) {}

DistributionFunction::~DistributionFunction() {
  for (std::pair<SubLattice, thrust_vectors> element : m_df) {
    delete element.second.gpu;
    delete element.second.cpu;
  }
}

void DistributionFunction::allocate(const SubLattice p) {
  int size = p.getArrayStride() * m_Q;
  m_df[p] = {.gpu = new thrust::device_vector<real>(size),
             .cpu = new thrust::host_vector<real>(size)};
}

std::vector<SubLattice> DistributionFunction::getAllocatedSubLattices() {
  std::vector<SubLattice> subLattices;
  for (std::pair<SubLattice, thrust_vectors> element : m_df) {
    subLattices.push_back(element.first);
  }
  return subLattices;
}

// Fill the ith array, i.e. the ith distribution function with a constant
// value for all nodes
void DistributionFunction::fill(unsigned int dfIdx, real value) {
  for (std::pair<SubLattice, thrust_vectors> element : m_df) {
    const int size = element.first.getArrayStride();
    thrust::device_vector<real>* dfGPU = element.second.gpu;
    thrust::fill(dfGPU->begin() + dfIdx * size,
                 dfGPU->begin() + (dfIdx + 1) * size, value);
    thrust::host_vector<real>* dfCPU = element.second.cpu;
    thrust::fill(dfCPU->begin() + dfIdx * size,
                 dfCPU->begin() + (dfIdx + 1) * size, value);
  }
}

// Read/write to allocated subLattices, excluding halos
real& DistributionFunction::operator()(unsigned int dfIdx, unsigned int x,
                                       unsigned int y, unsigned int z) {
  glm::ivec3 p(x, y, z);
  for (std::pair<SubLattice, thrust_vectors> element : m_df) {
    SubLattice subLattice = element.first;
    thrust_vectors vec = element.second;
    glm::ivec3 min = subLattice.getLatticeMin();
    glm::ivec3 max = subLattice.getLatticeMax();
    glm::ivec3 n = subLattice.getArrayDims();
    if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
        p.y < max.y && p.z < max.z && dfIdx < m_Q) {
      glm::ivec3 q = p - subLattice.getLatticeMin() + glm::ivec3(1, 1, 1);
      int idx = I4D(dfIdx, q.x, q.y, q.z, n.x, n.y, n.z);
      assert(vec.cpu->size() == n.x * n.y * n.z * m_Q);
      assert(idx < vec.cpu->size());
      return (*vec.cpu)[idx];
    }
  }
  throw std::out_of_range("Invalid range");
}

// Read/write to specific allocated subLattice, including halos
// start at -1 end at n + 1
real& DistributionFunction::operator()(SubLattice subLattice,
                                       unsigned int dfIdx, int x, int y,
                                       int z) {
  if (m_df.find(subLattice) == m_df.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::host_vector<real>* cpuVector = m_df.at(subLattice).cpu;
  int idx = subLattice.toLocalIndex(dfIdx, x, y, z);
  return (*cpuVector)[idx];
}

// Return a pointer to the beginning of the GPU memory
real* DistributionFunction::gpu_ptr(SubLattice subLattice, unsigned int dfIdx,
                                    int x, int y, int z, bool halo) {
  if (m_df.find(subLattice) == m_df.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<real>* gpuVector = m_df.at(subLattice).gpu;
  int idx;
  if (!halo) {
    idx = subLattice.toLocalIndex(dfIdx, x, y, z);
  } else {
    glm::ivec3 n = subLattice.getArrayDims();
    idx = I4D(dfIdx, x, y, z, n.x, n.y, n.z);
  }
  return thrust::raw_pointer_cast(&(*gpuVector)[idx]);
}

real* DistributionFunction::gpu_ptr(SubLattice subLattice, unsigned int idx) {
  if (m_df.find(subLattice) == m_df.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<real>* gpuVector = m_df.at(subLattice).gpu;
  return thrust::raw_pointer_cast(&(*gpuVector)[idx]);
}

void DistributionFunction::pushSubLattice(int srcDev, SubLattice subLattice,
                                          int dstDev,
                                          DistributionFunction* dstDf,
                                          cudaStream_t cpyStream) {
  size_t size = subLattice.getArrayStride() * m_Q * sizeof(real);
  real* srcPtr = gpu_ptr(subLattice);
  real* dstPtr = dstDf->gpu_ptr(subLattice);
  CUDA_RT_CALL(
      cudaMemcpyPeerAsync(dstPtr, dstDev, srcPtr, srcDev, size, cpyStream));
}

// Upload the distributions functions from the CPU to the GPU
DistributionFunction& DistributionFunction::upload() {
  for (std::pair<SubLattice, thrust_vectors> element : m_df) {
    *element.second.gpu = *element.second.cpu;
  }
  return *this;
}

// Download the distributions functions from the GPU to the CPU
DistributionFunction& DistributionFunction::download() {
  for (std::pair<SubLattice, thrust_vectors> element : m_df) {
    *element.second.cpu = *element.second.gpu;
  }
  return *this;
}

DistributionFunction& DistributionFunction::operator=(
    const DistributionFunction& f) {
  if (getLatticeDims() == f.getLatticeDims()) {
    for (std::pair<SubLattice, thrust_vectors> element : m_df) {
      SubLattice subLattice = element.first;
      thrust_vectors v1 = element.second;
      if (f.m_df.find(subLattice) != f.m_df.end()) {
        thrust_vectors v2 = f.m_df.at(subLattice);
        // thrust::copy(v2.gpu->begin(), v2.gpu->end(), v1.gpu->begin());
        thrust::copy(v2.cpu->begin(), v2.cpu->end(), v1.cpu->begin());
      } else {
        throw std::out_of_range(
            "RHS must have allocated all subLattices LHS has");
      }
    }
    return *this;
  }
  throw std::out_of_range("Distribution functions must have the same size");
}

// Static function to swap two DistributionFunctionsGroup
void DistributionFunction::swap(DistributionFunction* f1,
                                DistributionFunction* f2) {
  if (f1->m_df.size() == f2->m_df.size()) {
    for (std::pair<SubLattice, thrust_vectors> element : f1->m_df) {
      SubLattice subLattice = element.first;
      thrust_vectors v1 = element.second;
      if (f2->m_df.find(subLattice) != f2->m_df.end()) {
        thrust_vectors v2 = f2->m_df.at(subLattice);
        (*v1.gpu).swap(*v2.gpu);
        (*v1.cpu).swap(*v2.cpu);
      } else {
        throw std::out_of_range(
            "Cannot swap incompatible distribution functions");
      }
    }
    return;
  }
  throw std::out_of_range("Distribution functions must have the same size");
}

unsigned long DistributionFunction::memoryUse() {
  int sum = 0;
  for (std::pair<SubLattice, thrust_vectors> element : m_df)
    sum += element.second.cpu->size() * sizeof(real);
  return sum;
}

std::ostream& operator<<(std::ostream& os, DistributionFunction& df) {
  std::vector<SubLattice> subLattices = df.getSubLattices();
  glm::ivec3 pMax = df.getNumSubLattices();
  for (int q = 0; q < df.getQ(); q++) {
    for (int pz = 0; pz < pMax.z; pz++) {
      for (int py = 0; py < pMax.y; py++) {
        for (int px = 0; px < pMax.x; px++) {
          SubLattice subLattice = df.getSubLattice(px, py, pz);

          if (!df.isAllocated(subLattice)) continue;

          os << "q=" << q << ", subLattice=" << glm::ivec3(px, py, pz)
             << std::endl;

          glm::ivec3 min = subLattice.getLatticeMin() - glm::ivec3(1, 1, 1);
          glm::ivec3 max = subLattice.getLatticeMax() + glm::ivec3(1, 1, 1);
          for (int z = max.z - 1; z >= min.z; z--) {
            for (int y = max.y - 1; y >= min.y; y--) {
              for (int x = min.x; x < max.x; x++) {
                try {
                  os << std::setfill('0') << std::setw(2)
                     << df(subLattice, q, x, y, z);
                } catch (std::out_of_range& e) {
                  os << "X";
                }
                if (x < max.x - 1) os << ",";
              }
              os << std::endl;
            }
            os << std::endl;
          }
        }
      }
    }
  }
  return os;
}
