#include "DistributionArray.hpp"

DistributionArray::DistributionArray(unsigned int Q, unsigned int latticeSizeX,
                                     unsigned int latticeSizeY,
                                     unsigned int latticeSizeZ,
                                     unsigned int subdivisions,
                                     unsigned int haloSize)
    : DistributedLattice(latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions,
                         haloSize),
      m_Q(Q) {}

DistributionArray::~DistributionArray() {
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays) {
    delete element.second.gpu;
    delete element.second.cpu;
  }
}

void DistributionArray::allocate(const SubLattice p) {
  int size = p.getArrayStride() * m_Q;
  m_arrays[p] = {.gpu = new thrust::device_vector<real>(size),
                 .cpu = new thrust::host_vector<real>(size)};
}

std::vector<SubLattice> DistributionArray::getAllocatedSubLattices() {
  std::vector<SubLattice> subLattices;
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays)
    subLattices.push_back(element.first);
  return subLattices;
}

// Fill the ith array, i.e. the ith distribution function with a constant
// value for all nodes
void DistributionArray::fill(unsigned int dfIdx, real value) {
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays) {
    const int size = element.first.getArrayStride();
    thrust::device_vector<real>* gpuVec = element.second.gpu;
    thrust::fill(gpuVec->begin() + dfIdx * size,
                 gpuVec->begin() + (dfIdx + 1) * size, value);
    thrust::host_vector<real>* cpuVec = element.second.cpu;
    thrust::fill(cpuVec->begin() + dfIdx * size,
                 cpuVec->begin() + (dfIdx + 1) * size, value);
  }
}

void DistributionArray::exchange(SubLattice subLattice, DistributionArray* ndf,
                                 SubLattice neighbour, D3Q7::Enum direction,
                                 cudaStream_t stream) {
  SubLatticeSegment segment =
      getSubLatticeSegment(subLattice, neighbour, direction);

  for (int q : D3Q27ranks[direction]) {
    if (q >= getQ()) break;
    real* dfPtr = gpu_ptr(subLattice, q, segment.m_src.x, segment.m_src.y,
                          segment.m_src.z);
    real* ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x, segment.m_dst.y,
                                segment.m_dst.z);
    CUDA_RT_CALL(cudaMemcpy2DAsync(ndfPtr, segment.m_dstStride, dfPtr,
                                   segment.m_srcStride, segment.m_segmentLength,
                                   segment.m_numSegments, cudaMemcpyDefault,
                                   stream));
  }
}

// Read/write to specific allocated subLattice on CPU
real& DistributionArray::operator()(SubLattice subLattice, unsigned int dfIdx,
                                    int x, int y, int z) {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::host_vector<real>* cpuVec = m_arrays.at(subLattice).cpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(dfIdx, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return (*cpuVec)[idx];
}

void DistributionArray::resize(size_t size) {
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays) {
    thrust::device_vector<real>* gpuVec = element.second.gpu;
    thrust::host_vector<real>* cpuVec = element.second.cpu;
    gpuVec.erase(gpuVec.begin(), gpuVec.end());
    gpuVec.reserve(gpuVec->getSize());
    gpuVec.resize(gpuVec->getSize(), 0);
    cpuVec.erase(cpuVec.begin(), cpuVec.end());
    cpuVec.reserve(cpuVec->getSize());
    cpuVec.resize(cpuVec->getSize(), 0);
  }
}

// Return a pointer to the beginning of the GPU memory
real* DistributionArray::gpu_ptr(SubLattice subLattice, unsigned int dfIdx,
                                 int x, int y, int z) {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<real>* gpuVec = m_arrays.at(subLattice).gpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(dfIdx, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return thrust::raw_pointer_cast(&(*gpuVec)[idx]);
}

void DistributionArray::scatter(DistributionArray* src, SubLattice dstPart,
                                cudaStream_t stream) {
  SubLattice srcPart = src->getAllocatedSubLattices().at(0);

  glm::ivec3 dstLatDim = getLatticeDims();
  glm::ivec3 srcLatDim = src->getLatticeDims();
  glm::ivec3 srcDim = srcPart.getArrayDims();

  // Lattices must have same size
  if (srcLatDim.x != dstLatDim.x || srcLatDim.y != dstLatDim.y ||
      srcLatDim.z != dstLatDim.z || getQ() != src->getQ())
    throw std::out_of_range("Lattice sizes must be equal");

  // The source subLattice must have the size of the entire lattice
  if (srcLatDim.x != srcDim.x || srcLatDim.y != srcDim.y ||
      srcLatDim.z != srcDim.z)
    throw std::out_of_range(
        "Source sub lattice must have size of entire lattice");

  for (int q = 0; q < getQ(); q++) {
    glm::ivec3 srcPos = dstPart.getLatticeMin();
    glm::ivec3 dstPos = dstPart.getHalo();
    glm::ivec3 dstDim = dstPart.getArrayDims();
    glm::ivec3 cpyExt = dstPart.getLatticeDims();

    cudaMemcpy3DParms cpy = {0};
    // Source pointer
    cpy.srcPtr = make_cudaPitchedPtr(
        src->gpu_ptr(srcPart, q, srcPos.x, srcPos.y, srcPos.z),
        srcDim.x * sizeof(real), srcDim.x, srcDim.y);
    // Destination pointer
    cpy.dstPtr =
        make_cudaPitchedPtr(gpu_ptr(dstPart, q, dstPos.x, dstPos.y, dstPos.z),
                            dstDim.x * sizeof(real), dstDim.x, dstDim.y);
    // Extent of 3D copy
    cpy.extent = make_cudaExtent(cpyExt.x * sizeof(real), cpyExt.y, cpyExt.z);
    cpy.kind = cudaMemcpyDefault;

    cudaMemcpy3DAsync(&cpy, stream);
  }
}

void DistributionArray::gather(SubLattice srcPart, DistributionArray* dst,
                               cudaStream_t stream) {
  SubLattice dstPart = dst->getAllocatedSubLattices().at(0);

  glm::ivec3 srcLatDim = getLatticeDims();
  glm::ivec3 dstLatDim = dst->getLatticeDims();
  glm::ivec3 dstDim = dstPart.getArrayDims();

  // Lattices must have same size
  if (srcLatDim.x != dstLatDim.x || srcLatDim.y != dstLatDim.y ||
      srcLatDim.z != dstLatDim.z || getQ() != dst->getQ())
    throw std::out_of_range("Lattice sizes must be equal");

  // The destination subLattice must have the size of the entire lattice
  if (srcLatDim.x != dstDim.x || srcLatDim.y != dstDim.y ||
      srcLatDim.z != dstDim.z)
    throw std::out_of_range(
        "Destination sub lattice must have size of entire lattice");

  for (int q = 0; q < getQ(); q++) {
    glm::ivec3 srcPos = srcPart.getHalo();
    glm::ivec3 dstPos = srcPart.getLatticeMin();
    glm::ivec3 srcDim = srcPart.getArrayDims();
    glm::ivec3 cpyExt = srcPart.getLatticeDims();

    cudaMemcpy3DParms cpy = {0};
    // Source pointer
    cpy.srcPtr =
        make_cudaPitchedPtr(gpu_ptr(srcPart, q, srcPos.x, srcPos.y, srcPos.z),
                            srcDim.x * sizeof(real), srcDim.x, srcDim.y);
    // Destination pointer
    cpy.dstPtr = make_cudaPitchedPtr(
        dst->gpu_ptr(dstPart, q, dstPos.x, dstPos.y, dstPos.z),
        dstDim.x * sizeof(real), dstDim.x, dstDim.y);
    // Extent of 3D copy
    cpy.extent = make_cudaExtent(cpyExt.x * sizeof(real), cpyExt.y, cpyExt.z);
    cpy.kind = cudaMemcpyDefault;

    cudaMemcpy3DAsync(&cpy, stream);
  }
}

// Upload the distributions functions from the CPU to the GPU
DistributionArray& DistributionArray::upload() {
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays)
    *element.second.gpu = *element.second.cpu;
  return *this;
}

// Download the distributions functions from the GPU to the CPU
DistributionArray& DistributionArray::download() {
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays)
    *element.second.cpu = *element.second.gpu;
  return *this;
}

DistributionArray& DistributionArray::operator=(const DistributionArray& f) {
  if (getLatticeDims() == f.getLatticeDims()) {
    for (std::pair<SubLattice, thrust_vectors> element : m_arrays) {
      SubLattice subLattice = element.first;
      thrust_vectors v1 = element.second;
      if (f.m_arrays.find(subLattice) != f.m_arrays.end()) {
        thrust_vectors v2 = f.m_arrays.at(subLattice);
        // thrust::copy(v2.gpu->begin(), v2.gpu->end(), v1.gpu->begin());
        thrust::copy(v2.cpu->begin(), v2.cpu->end(), v1.cpu->begin());
      } else {
        throw std::out_of_range(
            "RHS must have allocated all subLattices of LHS");
      }
    }
    return *this;
  }
  throw std::out_of_range("Distribution functions must have the same size");
}

// Static function to swap two DistributionArraysGroup
void DistributionArray::swap(DistributionArray* f1, DistributionArray* f2) {
  if (f1->m_arrays.size() == f2->m_arrays.size()) {
    for (std::pair<SubLattice, thrust_vectors> element : f1->m_arrays) {
      SubLattice subLattice = element.first;
      thrust_vectors v1 = element.second;
      if (f2->m_arrays.find(subLattice) != f2->m_arrays.end()) {
        thrust_vectors v2 = f2->m_arrays.at(subLattice);
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

size_t DistributionArray::memoryUse() {
  int sum = 0;
  for (std::pair<SubLattice, thrust_vectors> element : m_arrays)
    sum += element.second.cpu->size() * sizeof(real);
  return sum;
}

std::ostream& operator<<(std::ostream& os, DistributionArray& df) {
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

          glm::ivec3 min = glm::ivec3(0, 0, 0);
          glm::ivec3 max =
              subLattice.getLatticeDims() + subLattice.getHalo() * 2;
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
