#include "DistributedDFGroup.hpp"

DistributedDFGroup::DistributedDFGroup(unsigned int Q,
                                       unsigned int latticeSizeX,
                                       unsigned int latticeSizeY,
                                       unsigned int latticeSizeZ,
                                       unsigned int subdivisions)
    : Topology(Q, latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions) {}

DistributedDFGroup::~DistributedDFGroup() {
  for (std::pair<Partition, thrust_vectors> element : m_df) {
    delete element.second.gpu;
    delete element.second.cpu;
  }
}

void DistributedDFGroup::allocate(Partition p) {
  int size = p.getArraySize() * m_Q;
  m_df[p] = {.gpu = new thrust::device_vector<real>(size),
             .cpu = new thrust::host_vector<real>(size)};
}

std::vector<Partition> DistributedDFGroup::getAllocatedPartitions() {
  std::vector<Partition> partitions;
  for (std::pair<Partition, thrust_vectors> element : m_df) {
    partitions.push_back(element.first);
  }
  return partitions;
}

// Fill the ith array, i.e. the ith distribution function with a constant
// value for all nodes
void DistributedDFGroup::fill(unsigned int dfIdx, real value) {
  for (std::pair<Partition, thrust_vectors> element : m_df) {
    const int size = element.first.getArraySize();
    thrust::device_vector<real>* dfGPU = element.second.gpu;
    thrust::fill(dfGPU->begin() + dfIdx * size,
                 dfGPU->begin() + (dfIdx + 1) * size, value);
    thrust::host_vector<real>* dfCPU = element.second.cpu;
    thrust::fill(dfCPU->begin() + dfIdx * size,
                 dfCPU->begin() + (dfIdx + 1) * size, value);
  }
}

// Read/write to allocated partitions, excluding halos
real& DistributedDFGroup::operator()(unsigned int dfIdx, unsigned int x,
                                     unsigned int y, unsigned int z) {
  glm::ivec3 p(x, y, z);
  for (std::pair<Partition, thrust_vectors> element : m_df) {
    Partition partition = element.first;
    thrust_vectors vec = element.second;
    glm::ivec3 min = partition.getLatticeMin();
    glm::ivec3 max = partition.getLatticeMax();
    glm::ivec3 n = partition.getArrayDims();
    if (p.x >= min.x && p.y >= min.y && p.z >= min.z && p.x < max.x &&
        p.y < max.y && p.z < max.z && dfIdx < m_Q) {
      glm::ivec3 q = p - partition.getLatticeMin() + glm::ivec3(1, 1, 1);
      int idx = I4D(dfIdx, q.x, q.y, q.z, n.x, n.y, n.z);
      assert(vec.cpu->size() == n.x * n.y * n.z * m_Q);
      assert(idx < vec.cpu->size());
      return (*vec.cpu)[idx];
    }
  }
  throw std::out_of_range("Invalid range");
}

// Read/write to specific allocated partition, including halos
// start at -1 end at n + 1
real& DistributedDFGroup::operator()(Partition partition, unsigned int dfIdx,
                                     int x, int y, int z) {
  if (m_df.find(partition) == m_df.end())
    throw std::out_of_range("Partition not allocated");
  thrust::host_vector<real>* cpuVector = m_df.at(partition).cpu;
  int idx = partition.toLocalIndex(dfIdx, x, y, z);
  return (*cpuVector)[idx];
}

// Return a pointer to the beginning of the GPU memory
real* DistributedDFGroup::gpu_ptr(Partition partition, unsigned int dfIdx,
                                  int x, int y, int z) {
  if (m_df.find(partition) == m_df.end())
    throw std::out_of_range("Partition not allocated");
  thrust::device_vector<real>* gpuVector = m_df.at(partition).gpu;
  int idx = partition.toLocalIndex(dfIdx, x, y, z);
  return thrust::raw_pointer_cast(&(*gpuVector)[idx]);
}

real* DistributedDFGroup::gpu_ptr(Partition partition, unsigned int idx) {
  if (m_df.find(partition) == m_df.end())
    throw std::out_of_range("Partition not allocated");
  thrust::device_vector<real>* gpuVector = m_df.at(partition).gpu;
  return thrust::raw_pointer_cast(&(*gpuVector)[idx]);
}

__global__ void HaloExchangeKernel(real* __restrict__ srcs,
                                   int* __restrict__ srcIdxs, int srcQStride,
                                   real* __restrict__ dsts,
                                   int* __restrict__ dstIdxs, int dstQStride,
                                   int numElems, int numQ) {
  if (threadIdx.x >= numQ || blockIdx.x >= numElems) return;
  const int srcIdx = srcIdxs[blockIdx.x] + threadIdx.x * srcQStride;
  const int dstIdx = dstIdxs[blockIdx.x] + threadIdx.x * dstQStride;
  dsts[dstIdx] = srcs[srcIdx];
}

void DistributedDFGroup::pushHaloFull(Partition partition, Partition neighbour,
                                      DistributedDFGroup* dstDf,
                                      cudaStream_t cpyStream) {
  HaloExchangeData* haloData = m_haloData[partition][neighbour];
  if (haloData->srcIndexH.size() == 0) return;

  if (haloData->srcIndexH.size() != haloData->srcIndexD.size())
    haloData->srcIndexD = thrust::device_vector<int>(haloData->srcIndexH);
  if (haloData->dstIndexH.size() != haloData->dstIndexD.size())
    haloData->dstIndexD = thrust::device_vector<int>(haloData->dstIndexH);

  assert(haloData->srcIndexH.size() == haloData->srcIndexD.size() &&
         haloData->srcIndexH.size() == haloData->dstIndexD.size() &&
         haloData->srcIndexH.size() == haloData->dstIndexH.size());

  int* srcIdxPtr = thrust::raw_pointer_cast(&(haloData->srcIndexD)[0]);
  int* dstIdxPtr = thrust::raw_pointer_cast(&(haloData->dstIndexD)[0]);

  real* srcPtr = gpu_ptr(partition);
  real* dstPtr = dstDf->gpu_ptr(neighbour);

  int srcQStride = partition.getArraySize();
  int dstQStride = neighbour.getArraySize();
  int numElems = haloData->srcIndexH.size();

  dim3 gridSize(numElems, 1, 1);
  dim3 blockSize(m_Q, 1, 1);

  HaloExchangeKernel<<<gridSize, blockSize, 0, cpyStream>>>(
      srcPtr, srcIdxPtr, srcQStride, dstPtr, dstIdxPtr, dstQStride, numElems,
      m_Q);
  // CUDA_CHECK_ERRORS("HaloExchangeKernel");
}

// void DistributedDFGroup::pushHaloReduced(Partition partition,
//                                          DistributedDFGroup* dstDf,
//                                          cudaStream_t cpyStream) {
//   HaloExchangeData* haloData = m_haloData[partition].at(Q);
//   if (haloData->srcIndexH.size() == 0) return;
//   Partition neighbour = haloData->neighbour;

//   if (haloData->srcIndexH.size() != haloData->srcIndexD.size())
//     haloData->srcIndexD = thrust::device_vector<int>(haloData->srcIndexH);
//   if (haloData->dstIndexH.size() != haloData->dstIndexD.size())
//     haloData->dstIndexD = thrust::device_vector<int>(haloData->dstIndexH);

//   assert(haloData->srcIndexH.size() == haloData->srcIndexD.size() &&
//          haloData->srcIndexH.size() == haloData->dstIndexD.size() &&
//          haloData->srcIndexH.size() == haloData->dstIndexH.size());

//   int* srcIdxPtr = thrust::raw_pointer_cast(&(haloData->srcIndexD)[0]);
//   int* dstIdxPtr = thrust::raw_pointer_cast(&(haloData->dstIndexD)[0]);

//   real* srcPtr = gpu_ptr(partition);
//   real* dstPtr = dstDf->gpu_ptr(neighbour);

//   int srcQStride = partition.getArraySize();
//   int dstQStride = neighbour.getArraySize();
//   int numElems = haloData->srcIndexH.size();

//   dim3 gridSize(numElems, 1, 1);
//   dim3 blockSize(m_Q, 1, 1);

//   HaloExchangeKernel<<<gridSize, blockSize, 0, cpyStream>>>(
//       srcPtr, srcIdxPtr, srcQStride, dstPtr, dstIdxPtr, dstQStride, numElems,
//       Q);
//   CUDA_CHECK_ERRORS("HaloExchangeKernel");
// }

void DistributedDFGroup::pushPartition(int srcDev, Partition partition,
                                       int dstDev, DistributedDFGroup* dstDf,
                                       cudaStream_t cpyStream) {
  size_t size = partition.getArraySize() * m_Q * sizeof(real);
  real* srcPtr = gpu_ptr(partition);
  real* dstPtr = dstDf->gpu_ptr(partition);
  CUDA_RT_CALL(
      cudaMemcpyPeerAsync(dstPtr, dstDev, srcPtr, srcDev, size, cpyStream));
}

// Upload the distributions functions from the CPU to the GPU
DistributedDFGroup& DistributedDFGroup::upload() {
  for (std::pair<Partition, thrust_vectors> element : m_df) {
    *element.second.gpu = *element.second.cpu;
  }
  return *this;
}

// Download the distributions functions from the GPU to the CPU
DistributedDFGroup& DistributedDFGroup::download() {
  for (std::pair<Partition, thrust_vectors> element : m_df) {
    *element.second.cpu = *element.second.gpu;
  }
  return *this;
}

DistributedDFGroup& DistributedDFGroup::operator=(const DistributedDFGroup& f) {
  if (getLatticeDims() == f.getLatticeDims()) {
    for (std::pair<Partition, thrust_vectors> element : m_df) {
      Partition partition = element.first;
      thrust_vectors v1 = element.second;
      if (f.m_df.find(partition) != f.m_df.end()) {
        thrust_vectors v2 = f.m_df.at(partition);
        // thrust::copy(v2.gpu->begin(), v2.gpu->end(), v1.gpu->begin());
        thrust::copy(v2.cpu->begin(), v2.cpu->end(), v1.cpu->begin());
      } else {
        throw std::out_of_range(
            "RHS must have allocated all partitions LHS has");
      }
    }
    return *this;
  }
  throw std::out_of_range("Distribution functions must have the same size");
}

// Static function to swap two DistributionFunctionsGroup
void DistributedDFGroup::swap(DistributedDFGroup* f1, DistributedDFGroup* f2) {
  if (f1->m_df.size() == f2->m_df.size()) {
    for (std::pair<Partition, thrust_vectors> element : f1->m_df) {
      Partition partition = element.first;
      thrust_vectors v1 = element.second;
      if (f2->m_df.find(partition) != f2->m_df.end()) {
        thrust_vectors v2 = f2->m_df.at(partition);
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

unsigned long DistributedDFGroup::memoryUse() {
  int sum = 0;
  for (std::pair<Partition, thrust_vectors> element : m_df)
    sum += element.second.cpu->size() * sizeof(real);
  return sum;
}

std::ostream& operator<<(std::ostream& os, DistributedDFGroup& df) {
  std::vector<Partition*> partitions = df.getPartitions();
  glm::ivec3 pMax = df.getNumPartitions();
  for (int q = 0; q < df.getQ(); q++) {
    for (int pz = 0; pz < pMax.z; pz++) {
      for (int py = 0; py < pMax.y; py++) {
        for (int px = 0; px < pMax.x; px++) {
          Partition* partition = df.getPartition(px, py, pz);
          os << "q=" << q << ", partition=" << glm::ivec3(px, py, pz)
             << std::endl;

          glm::ivec3 min = partition->getLatticeMin() - glm::ivec3(1, 1, 1);
          glm::ivec3 max = partition->getLatticeMax() + glm::ivec3(1, 1, 1);
          for (int z = min.z; z < max.z; z++) {
            for (int y = min.y; y < max.y; y++) {
              for (int x = min.x; x < max.x; x++) {
                try {
                  os << df(*partition, q, x, y, z);
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
