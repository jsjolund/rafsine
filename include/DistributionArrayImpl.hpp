#pragma once
#include "DistributionArray.hpp"

template <class T>
DistributionArray<T>::DistributionArray(unsigned int q, unsigned int nx,
                                        unsigned int ny, unsigned int nz,
                                        unsigned int nd, unsigned int haloSize)
    : DistributedLattice(nx, ny, nz, nd, haloSize), m_Q(q) {}

template <class T>
DistributionArray<T>::~DistributionArray() {
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays) {
    delete element.second->gpu;
    delete element.second->cpu;
  }
}

template <class T>
void DistributionArray<T>::allocate(SubLattice subLattice) {
  if (subLattice.isEmpty())
    subLattice = getSubLattice(0, 0, 0);
  else if (m_arrays.find(subLattice) != m_arrays.end())
    throw std::out_of_range("SubLattice already allocated");
  int size = subLattice.getArrayStride() * m_Q;
  m_arrays[subLattice] = new MemoryStore(size);
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
void DistributionArray<T>::fill(unsigned int q, T value) {
  for (std::pair<SubLattice, MemoryStore*> element : m_arrays) {
    const int size = element.first.getArrayStride();
    thrust::device_vector<T>* gpuVec = element.second->gpu;
    thrust::fill(gpuVec->begin() + q * size, gpuVec->begin() + (q + 1) * size,
                 value);
    thrust::host_vector<T>* cpuVec = element.second->cpu;
    thrust::fill(cpuVec->begin() + q * size, cpuVec->begin() + (q + 1) * size,
                 value);
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
T& DistributionArray<T>::operator()(SubLattice subLattice, unsigned int q,
                                    int x, int y, int z) {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::host_vector<T>* cpuVec = m_arrays.at(subLattice)->cpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(q, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return (*cpuVec)[idx];
}

// Read only, from specific allocated subLattice on CPU
template <class T>
T DistributionArray<T>::read(SubLattice subLattice, unsigned int q, int x,
                             int y, int z) const {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::host_vector<T>* cpuVec = m_arrays.at(subLattice)->cpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(q, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return (*cpuVec)[idx];
}

template <class T>
void DistributionArray<T>::getMin(SubLattice subLattice, T* min) const {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(subLattice)->gpu;
  if (gpuVec->size() == 0) return;
  auto input_end =
      thrust::remove_if(gpuVec->begin(), gpuVec->end(), CUDA_isNaN());
  *min = *thrust::min_element(gpuVec->begin(), input_end);
}

template <class T>
void DistributionArray<T>::getMax(SubLattice subLattice, T* max) const {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(subLattice)->gpu;
  if (gpuVec->size() == 0) return;
  auto input_end =
      thrust::remove_if(gpuVec->begin(), gpuVec->end(), CUDA_isNaN());
  *max = *thrust::max_element(gpuVec->begin(), input_end);
}

// Return a pointer to the beginning of the GPU memory
template <class T>
T* DistributionArray<T>::gpu_ptr(SubLattice subLattice, unsigned int q, int x,
                                 int y, int z) const {
  if (m_arrays.find(subLattice) == m_arrays.end())
    throw std::out_of_range("SubLattice not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(subLattice)->gpu;
  glm::ivec3 srcLatDim = subLattice.getArrayDims();
  int idx = I4D(q, x, y, z, srcLatDim.x, srcLatDim.y, srcLatDim.z);
  return thrust::raw_pointer_cast(&(*gpuVec)[idx]);
}

template <class T>
void DistributionArray<T>::scatter(const DistributionArray<T>& src,
                                   SubLattice dstPart, cudaStream_t stream) {
  SubLattice srcPart = src.getSubLattice(0, 0, 0);

  glm::ivec3 dstLatDim = getLatticeDims();
  glm::ivec3 srcLatDim = src.getLatticeDims();
  glm::ivec3 srcDim = srcPart.getArrayDims();

  // Lattices must have same size
  if (srcLatDim.x != dstLatDim.x || srcLatDim.y != dstLatDim.y ||
      srcLatDim.z != dstLatDim.z || getQ() != src.getQ())
    throw std::out_of_range("Lattice sizes must be equal");

  // The source subLattice must have the size of the entire lattice
  if (srcLatDim.x != srcDim.x || srcLatDim.y != srcDim.y ||
      srcLatDim.z != srcDim.z)
    throw std::out_of_range(
        "Source sub lattice must have size of entire lattice");

  glm::ivec3 srcPos = dstPart.getLatticeMin();
  glm::ivec3 dstPos = dstPart.getHalo();
  glm::ivec3 dstDim = dstPart.getArrayDims();
  glm::ivec3 cpyExt = dstPart.getLatticeDims();

  for (int q = 0; q < getQ(); q++) {
    memcpy3DAsync(src, srcPart, q, srcPos, srcDim, this, dstPart, q, dstPos,
                  dstDim, cpyExt, stream);
  }
}

template <class T>
void DistributionArray<T>::gather(SubLattice srcPart, DistributionArray<T>* dst,
                                  cudaStream_t stream) {
  // Lattices must have same number of 3D arrays
  if (getQ() != dst->getQ())
    throw std::out_of_range("Lattice sizes must be equal");
  for (int q = 0; q < getQ(); q++) gather(q, q, srcPart, dst, stream);
}

template <class T>
void DistributionArray<T>::gather(int srcQ, int dstQ, SubLattice srcPart,
                                  DistributionArray<T>* dst,
                                  cudaStream_t stream) {
  SubLattice dstPart = dst->getAllocatedSubLattices().at(0);

  glm::ivec3 srcLatDim = getLatticeDims();
  glm::ivec3 dstLatDim = dst->getLatticeDims();
  glm::ivec3 dstDim = dstPart.getArrayDims();
  // Lattices must have same size
  if (srcLatDim != dstLatDim)
    throw std::out_of_range("Lattice sizes must be equal");
  // The destination partition must have the size of the entire lattice
  if (srcLatDim != dstDim)
    throw std::out_of_range(
        "Destination sub lattice must have size of entire lattice");
  // Offset source position to exclude halos from copy
  glm::ivec3 srcPos = srcPart.getHalo();
  // The destination is the global position of the source partition
  glm::ivec3 dstPos = srcPart.getLatticeMin();
  // Dimensions of source parition must include halos
  glm::ivec3 srcDim = srcPart.getArrayDims();
  // Copy the full extent of the source partition, excluding halos
  glm::ivec3 cpyExt = srcPart.getLatticeDims();
  memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                dstPos, dstDim, cpyExt, stream);
}

template <class T>
void DistributionArray<T>::gatherSlice(glm::ivec3 slicePos, int srcQ, int dstQ,
                                       SubLattice srcPart,
                                       DistributionArray<T>* dst,
                                       cudaStream_t stream) {
  glm::ivec3 offset = slicePos - srcPart.getLatticeMin();

  SubLattice dstPart = dst->getAllocatedSubLattices().at(0);
  glm::ivec3 srcLatDim = getLatticeDims();
  glm::ivec3 dstLatDim = dst->getLatticeDims();
  glm::ivec3 dstDim = dstPart.getArrayDims();

  // Lattices must have same size
  if (srcLatDim != dstLatDim)
    throw std::out_of_range("Lattice sizes must be equal");

  // The destination subLattice must have the size of the entire lattice
  if (srcLatDim != dstDim)
    throw std::out_of_range(
        "Destination sub lattice must have size of entire lattice");

  if (slicePos.x >= srcPart.getLatticeMin().x &&
      slicePos.x < srcPart.getLatticeMax().x) {
    // Offset source position to exclude halos from copy
    glm::ivec3 srcPos = srcPart.getHalo();
    srcPos.x += offset.x;
    // The destination is the global position of the source partition
    glm::ivec3 dstPos = srcPart.getLatticeMin();
    dstPos.x = slicePos.x;
    // Dimensions of source parition must include halos
    glm::ivec3 srcDim = srcPart.getArrayDims();
    // Copy the full extent of the source partition, excluding halos
    glm::ivec3 cpyExt = srcPart.getLatticeDims();
    cpyExt.x = 1;
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
  }

  if (slicePos.y >= srcPart.getLatticeMin().y &&
      slicePos.y < srcPart.getLatticeMax().y) {
    // Offset source position to exclude halos from copy
    glm::ivec3 srcPos = srcPart.getHalo();
    srcPos.y += offset.y;
    // The destination is the global position of the source partition
    glm::ivec3 dstPos = srcPart.getLatticeMin();
    dstPos.y = slicePos.y;
    // Dimensions of source parition must include halos
    glm::ivec3 srcDim = srcPart.getArrayDims();
    // Copy the full extent of the source partition, excluding halos
    glm::ivec3 cpyExt = srcPart.getLatticeDims();
    cpyExt.y = 1;
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
  }

  if (slicePos.z >= srcPart.getLatticeMin().z &&
      slicePos.z < srcPart.getLatticeMax().z) {
    // Offset source position to exclude halos from copy
    glm::ivec3 srcPos = srcPart.getHalo();
    srcPos.z += offset.z;
    // The destination is the global position of the source partition
    glm::ivec3 dstPos = srcPart.getLatticeMin();
    dstPos.z = slicePos.z;
    // Dimensions of source parition must include halos
    glm::ivec3 srcDim = srcPart.getArrayDims();
    // Copy the full extent of the source partition, excluding halos
    glm::ivec3 cpyExt = srcPart.getLatticeDims();
    cpyExt.z = 1;
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
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
    for (std::pair<SubLattice, MemoryStore*> element : m_arrays) {
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
