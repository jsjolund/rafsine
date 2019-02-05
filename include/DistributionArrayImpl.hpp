#include "DistributionArray.hpp"

template <class T>
DistributionArray<T>::DistributionArray(
    unsigned int Q, unsigned int latticeSizeX, unsigned int latticeSizeY,
    unsigned int latticeSizeZ, unsigned int subdivisions, unsigned int haloSize)
    : DistributedLattice(latticeSizeX, latticeSizeY, latticeSizeZ, subdivisions,
                         haloSize),
      m_Q(Q) {}

template <class T>
DistributionArray<T>::~DistributionArray() {
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays) {
    delete element.second->gpu;
    delete element.second->cpu;
  }
}

template <class T>
void DistributionArray<T>::allocate(const SubLattice p) {
  int size = p.getArrayStride() * m_Q;
  m_arrays[p] = new MemoryStore(size);
}

template <class T>
std::vector<SubLattice> DistributionArray<T>::getAllocatedSubLattices() {
  std::vector<SubLattice> subLattices;
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays)
    subLattices.push_back(element.first);
  return subLattices;
}

// Fill the ith array, i.e. the ith distribution function with a constant
// value for all nodes
template <class T>
void DistributionArray<T>::fill(unsigned int dfIdx, T value) {
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays) {
    const int size = element.first.getArrayStride();
    thrust::device_vector<T>* gpuVec = element.second->gpu;
    thrust::fill(gpuVec->begin() + dfIdx * size,
                 gpuVec->begin() + (dfIdx + 1) * size, value);
    thrust::host_vector<T>* cpuVec = element.second->cpu;
    thrust::fill(cpuVec->begin() + dfIdx * size,
                 cpuVec->begin() + (dfIdx + 1) * size, value);
  }
}

template <class T>
void DistributionArray<T>::exchange(SubLattice subLattice,
                                    DistributionArray<T>* ndf,
                                    SubLattice neighbour, D3Q7::Enum direction,
                                    cudaStream_t stream) {
  SubLatticeSegment segment =
      getSubLatticeSegment(subLattice, neighbour, direction);

  for (int q : D3Q27ranks[direction]) {
    if (q >= getQ()) break;
    T* dfPtr = gpu_ptr(subLattice, q, segment.m_src.x, segment.m_src.y,
                       segment.m_src.z);
    T* ndfPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x, segment.m_dst.y,
                             segment.m_dst.z);
    CUDA_RT_CALL(cudaMemcpy2DAsync(ndfPtr, segment.m_dstStride, dfPtr,
                                   segment.m_srcStride, segment.m_segmentLength,
                                   segment.m_numSegments, cudaMemcpyDefault,
                                   stream));
  }
}

// Read/write to specific allocated subLattice on CPU
template <class T>
T& DistributionArray<T>::operator()(SubLattice subLattice, unsigned int dfIdx,
                                    int x, int y, int z) {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::host_vector<T>* cpuVec = m_arrays.at(subLattice)->cpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(dfIdx, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return (*cpuVec)[idx];
}

template <class T>
void DistributionArray<T>::getMinMax(SubLattice subLattice, int* min,
                                     int* max) {
  thrust::device_vector<T>* gpuVec = m_arrays[subLattice]->gpu;
  if (gpuVec->size() == 0) {
    *min = 0;
    *max = 0;
    return;
  }
  // Filter out NaN values
  auto input_end =
      thrust::remove_if(gpuVec->begin(), gpuVec->end(), CUDA_isNaN());
  typename thrust::device_vector<T>::iterator iter;
  iter = thrust::min_element(gpuVec->begin(), input_end);
  *min = *iter;
  iter = thrust::max_element(gpuVec->begin(), input_end);
  *max = *iter;
}

// Return a pointer to the beginning of the GPU memory
template <class T>
T* DistributionArray<T>::gpu_ptr(SubLattice subLattice, unsigned int dfIdx,
                                 int x, int y, int z) {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(subLattice)->gpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(dfIdx, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return thrust::raw_pointer_cast(&(*gpuVec)[idx]);
}

template <class T>
void DistributionArray<T>::scatter(DistributionArray<T>* src,
                                   SubLattice dstPart, cudaStream_t stream) {
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
        srcDim.x * sizeof(T), srcDim.x, srcDim.y);
    // Destination pointer
    cpy.dstPtr =
        make_cudaPitchedPtr(gpu_ptr(dstPart, q, dstPos.x, dstPos.y, dstPos.z),
                            dstDim.x * sizeof(T), dstDim.x, dstDim.y);
    // Extent of 3D copy
    cpy.extent = make_cudaExtent(cpyExt.x * sizeof(T), cpyExt.y, cpyExt.z);
    cpy.kind = cudaMemcpyDefault;

    cudaMemcpy3DAsync(&cpy, stream);
  }
}

template <class T>
void DistributionArray<T>::gather(SubLattice srcPart, DistributionArray<T>* dst,
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
                            srcDim.x * sizeof(T), srcDim.x, srcDim.y);
    // Destination pointer
    cpy.dstPtr = make_cudaPitchedPtr(
        dst->gpu_ptr(dstPart, q, dstPos.x, dstPos.y, dstPos.z),
        dstDim.x * sizeof(T), dstDim.x, dstDim.y);
    // Extent of 3D copy
    cpy.extent = make_cudaExtent(cpyExt.x * sizeof(T), cpyExt.y, cpyExt.z);
    cpy.kind = cudaMemcpyDefault;

    cudaMemcpy3DAsync(&cpy, stream);
  }
}

// Upload the distributions functions from the CPU to the GPU
template <class T>
DistributionArray<T>& DistributionArray<T>::upload() {
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays)
    *element.second->gpu = *element.second->cpu;
  return *this;
}

// Download the distributions functions from the GPU to the CPU
template <class T>
DistributionArray<T>& DistributionArray<T>::download() {
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays)
    *element.second->cpu = *element.second->gpu;
  return *this;
}

template <class T>
DistributionArray<T>& DistributionArray<T>::operator=(
    const DistributionArray<T>& f) {
  if (getLatticeDims() == f.getLatticeDims()) {
    for (std::pair<SubLattice, MemoryStore> element : m_arrays) {
      SubLattice subLattice = element.first;
      MemoryStore* v1 = element.second;
      if (f.m_arrays.find(subLattice) != f.m_arrays.end()) {
        MemoryStore* v2 = f.m_arrays.at(subLattice);
        // thrust::copy(v2.gpu->begin(), v2.gpu->end(), v1.gpu->begin());
        thrust::copy(v2->cpu->begin(), v2->cpu->end(), v1->cpu->begin());
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
template <class T>
void DistributionArray<T>::swap(DistributionArray<T>* f1,
                                DistributionArray<T>* f2) {
  if (f1->m_arrays.size() == f2->m_arrays.size()) {
    for (std::pair<SubLattice, MemoryStore*> element : f1->m_arrays) {
      SubLattice subLattice = element.first;
      MemoryStore* v1 = element.second;
      if (f2->m_arrays.find(subLattice) != f2->m_arrays.end()) {
        MemoryStore* v2 = f2->m_arrays.at(subLattice);
        (*v1->gpu).swap(*v2->gpu);
        (*v1->cpu).swap(*v2->cpu);
      } else {
        throw std::out_of_range(
            "Cannot swap incompatible distribution functions");
      }
    }
    return;
  }
  throw std::out_of_range("Distribution functions must have the same size");
}
