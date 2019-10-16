#include "DistributionArray.hpp"

template <class T>
DistributionArray<T>::MemoryStore::MemoryStore(size_t size) {
  gpu = new thrust::device_vector<T>(size);
  cpu = new thrust::host_vector<T>(size);
}

template <class T>
void DistributionArray<T>::memcpy3DAsync(const DistributionArray<T>& src,
                                         Partition srcPart,
                                         int srcQ,
                                         Eigen::Vector3i srcPos,
                                         Eigen::Vector3i srcDim,
                                         DistributionArray<T>* dst,
                                         Partition dstPart,
                                         int dstQ,
                                         Eigen::Vector3i dstPos,
                                         Eigen::Vector3i dstDim,
                                         Eigen::Vector3i cpyExt,
                                         cudaStream_t stream) {
  cudaMemcpy3DParms cpy = {0};
  // Source pointer
  cpy.srcPtr = make_cudaPitchedPtr(
      src.gpu_ptr(srcPart, srcQ, srcPos.x(), srcPos.y(), srcPos.z()),
      srcDim.x() * sizeof(T), srcDim.x(), srcDim.y());
  // Destination pointer
  cpy.dstPtr = make_cudaPitchedPtr(
      dst->gpu_ptr(dstPart, dstQ, dstPos.x(), dstPos.y(), dstPos.z()),
      dstDim.x() * sizeof(T), dstDim.x(), dstDim.y());
  // Extent of 3D copy
  cpy.extent = make_cudaExtent(cpyExt.x() * sizeof(T), cpyExt.y(), cpyExt.z());
  cpy.kind = cudaMemcpyDefault;

  CUDA_RT_CALL(cudaMemcpy3DAsync(&cpy, stream));
}

template <class T>
DistributionArray<T>::DistributionArray(unsigned int q,
                                        unsigned int nx,
                                        unsigned int ny,
                                        unsigned int nz,
                                        unsigned int nd,
                                        unsigned int ghostLayerSize)
    : DistributedLattice(nx, ny, nz, nd, ghostLayerSize), m_Q(q) {}

template <class T>
DistributionArray<T>::~DistributionArray() {
  for (std::pair<Partition, MemoryStore*> element : m_arrays) {
    if (element.second->gpu) delete element.second->gpu;
    if (element.second->cpu) delete element.second->cpu;
  }
}

template <class T>
void DistributionArray<T>::deallocate(MemoryType type, Partition partition) {
  if (partition.isEmpty()) partition = getPartition(0, 0, 0);
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  MemoryStore* store = m_arrays[partition];
  if (type == DEVICE_MEMORY) {
    delete store->gpu;
    store->gpu = NULL;
  } else {
    delete store->cpu;
    store->cpu = NULL;
  }
}

template <class T>
void DistributionArray<T>::allocate(Partition partition) {
  if (partition.isEmpty()) partition = getPartition(0, 0, 0);
  if (m_arrays.find(partition) != m_arrays.end())
    throw std::out_of_range("Partition already allocated");
  int size = partition.getArrayStride() * m_Q;
  m_arrays[partition] = new MemoryStore(size);
}

template <class T>
std::vector<Partition> DistributionArray<T>::getAllocatedPartitions() {
  std::vector<Partition> partitions;
  for (std::pair<Partition, MemoryStore*> element : m_arrays)
    partitions.push_back(element.first);
  return partitions;
}

template <class T>
T DistributionArray<T>::getAverage(Partition partition,
                                   unsigned int q,
                                   unsigned int offset,
                                   unsigned int length,
                                   T divisor) {
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  const int size = partition.getArrayStride();
  // download();
  // thrust::host_vector<T>* vec = m_arrays.at(partition)->cpu;
  // thrust::copy(vec->begin() + q * size, vec->begin() + (q + 1) * size,
  //              std::ostream_iterator<T>(std::cout, " "));
  // std::cout << std::endl;
  thrust::device_vector<T>* gpuVec = m_arrays.at(partition)->gpu;
  auto begin = gpuVec->begin() + q * size + offset;
  auto end = gpuVec->begin() + q * size + offset + length;
  return thrust::transform_reduce(
      begin, end, DistributionArray::division(static_cast<T>(length) * divisor),
      static_cast<T>(0), thrust::plus<T>());
}

// Fill the distribution function with a constant value for all nodes
template <class T>
void DistributionArray<T>::fill(T value, cudaStream_t stream) {
  for (std::pair<Partition, MemoryStore*> element : m_arrays) {
    const int size = element.first.getArrayStride();
    thrust::device_vector<T>* gpuVec = element.second->gpu;
    thrust::fill(thrust::cuda::par.on(stream), gpuVec->begin(), gpuVec->end(),
                 value);
    // thrust::host_vector<T>* cpuVec = element.second->cpu;
    // thrust::fill(cpuVec->begin(), cpuVec->end(), value);
  }
}

template <class T>
void DistributionArray<T>::exchange(Partition partition,
                                    DistributionArray<T>* ndf,
                                    Partition neighbour,
                                    D3Q7::Enum direction,
                                    cudaStream_t stream) {
  GhostLayerParameters segment = getGhostLayer(partition, neighbour, direction);

  for (int q : D3Q27ranks[direction]) {
    if (q >= getQ()) break;
    T* srcPtr = gpu_ptr(partition, q, segment.m_src.x(), segment.m_src.y(),
                        segment.m_src.z());
    T* dstPtr = ndf->gpu_ptr(neighbour, q, segment.m_dst.x(), segment.m_dst.y(),
                             segment.m_dst.z());
    CUDA_RT_CALL(cudaMemcpy2DAsync(
        dstPtr, segment.m_dpitch, srcPtr, segment.m_spitch, segment.m_width,
        segment.m_height, cudaMemcpyDefault, stream));
  }
}

// Read/write to specific allocated partition on CPU
template <class T>
T& DistributionArray<T>::operator()(Partition partition,
                                    unsigned int q,
                                    int x,
                                    int y,
                                    int z) {
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  thrust::host_vector<T>* cpuVec = m_arrays.at(partition)->cpu;
  Eigen::Vector3i srcLatDim = partition.getArrayExtents();
  int idx = I4D(q, x, y, z, srcLatDim.x(), srcLatDim.y(), srcLatDim.z());
  return (*cpuVec)[idx];
}

// Read only, from specific allocated partition on CPU
template <class T>
T DistributionArray<T>::read(Partition partition,
                             unsigned int q,
                             int x,
                             int y,
                             int z) const {
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  thrust::host_vector<T>* cpuVec = m_arrays.at(partition)->cpu;
  Eigen::Vector3i srcLatDim = partition.getArrayExtents();
  int idx = I4D(q, x, y, z, srcLatDim.x(), srcLatDim.y(), srcLatDim.z());
  return (*cpuVec)[idx];
}

template <class T>
T DistributionArray<T>::getMin(Partition partition) const {
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(partition)->gpu;
  auto input_end =
      thrust::remove_if(gpuVec->begin(), gpuVec->end(), CUDA_isNaN());
  return *thrust::min_element(gpuVec->begin(), input_end);
}

template <class T>
T DistributionArray<T>::getMax(Partition partition) const {
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(partition)->gpu;
  auto input_end =
      thrust::remove_if(gpuVec->begin(), gpuVec->end(), CUDA_isNaN());
  return *thrust::max_element(gpuVec->begin(), input_end);
}

// Return a pointer to the beginning of the GPU memory
template <class T>
T* DistributionArray<T>::gpu_ptr(Partition partition,
                                 unsigned int q,
                                 int x,
                                 int y,
                                 int z) const {
  if (m_arrays.find(partition) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  thrust::device_vector<T>* gpuVec = m_arrays.at(partition)->gpu;
  Eigen::Vector3i srcLatDim = partition.getArrayExtents();
  int idx = I4D(q, x, y, z, srcLatDim.x(), srcLatDim.y(), srcLatDim.z());
  return thrust::raw_pointer_cast(&(*gpuVec)[idx]);
}

template <class T>
void DistributionArray<T>::scatter(const DistributionArray<T>& src,
                                   Partition dstPart,
                                   cudaStream_t stream) {
  Partition srcPart = src.getPartition(0, 0, 0);

  Eigen::Vector3i dstLatDim = getExtents();
  Eigen::Vector3i srcLatDim = src.getExtents();
  Eigen::Vector3i srcDim = srcPart.getArrayExtents();

  // Lattices must have same size
  if (srcLatDim.x() != dstLatDim.x() || srcLatDim.y() != dstLatDim.y() ||
      srcLatDim.z() != dstLatDim.z() || getQ() != src.getQ())
    throw std::out_of_range("Lattice sizes must be equal");

  // The source partition must have the size of the entire lattice
  if (srcLatDim.x() != srcDim.x() || srcLatDim.y() != srcDim.y() ||
      srcLatDim.z() != srcDim.z())
    throw std::out_of_range(
        "Source sub lattice must have size of entire lattice");

  Eigen::Vector3i srcPos = dstPart.getMin();
  Eigen::Vector3i dstPos = dstPart.getGhostLayer();
  Eigen::Vector3i dstDim = dstPart.getArrayExtents();
  Eigen::Vector3i cpyExt = dstPart.getExtents();

  for (int q = 0; q < getQ(); q++) {
    memcpy3DAsync(src, srcPart, q, srcPos, srcDim, this, dstPart, q, dstPos,
                  dstDim, cpyExt, stream);
  }
}

template <class T>
void DistributionArray<T>::gather(Partition srcPart,
                                  DistributionArray<T>* dst,
                                  cudaStream_t stream) {
  // Lattices must have same number of 3D arrays
  if (getQ() != dst->getQ())
    throw std::out_of_range("Lattice sizes must be equal");
  for (int q = 0; q < getQ(); q++) gather(q, q, srcPart, dst, stream);
}

template <class T>
void DistributionArray<T>::gather(int srcQ,
                                  int dstQ,
                                  Partition srcPart,
                                  DistributionArray<T>* dst,
                                  cudaStream_t stream) {
  Partition dstPart = dst->getAllocatedPartitions().at(0);

  Eigen::Vector3i srcLatDim = getExtents();
  Eigen::Vector3i dstLatDim = dst->getExtents();
  Eigen::Vector3i dstDim = dstPart.getArrayExtents();
  // Lattices must have same size
  if (srcLatDim != dstLatDim)
    throw std::out_of_range("Lattice sizes must be equal");
  // The destination partition must have the size of the entire lattice
  if (srcLatDim != dstDim)
    throw std::out_of_range(
        "Destination sub lattice must have size of entire lattice");
  // Offset source position to exclude ghostLayers from copy
  Eigen::Vector3i srcPos = srcPart.getGhostLayer();
  // The destination is the global position of the source partition
  Eigen::Vector3i dstPos = srcPart.getMin();
  // Dimensions of source parition must include ghostLayers
  Eigen::Vector3i srcDim = srcPart.getArrayExtents();
  // Copy the full extent of the source partition, excluding ghostLayers
  Eigen::Vector3i cpyExt = srcPart.getExtents();
  memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                dstPos, dstDim, cpyExt, stream);
}

template <class T>
void DistributionArray<T>::gather(Eigen::Vector3i globalMin,
                                  Eigen::Vector3i globalMax,
                                  int srcQ,
                                  int dstQ,
                                  Partition srcPart,
                                  DistributionArray<T>* dst,
                                  Partition dstPart,
                                  cudaStream_t stream) {
  if (m_arrays.find(srcPart) == m_arrays.end())
    throw std::out_of_range("Partition not allocated");
  Eigen::Vector3i min, max;
  const int numVoxels = srcPart.intersect(globalMin, globalMax, &min, &max);
  // Size of the intersection
  const Eigen::Vector3i cpyExt = max - min;
  // Local position in partition
  const Eigen::Vector3i srcPos = min - srcPart.getMin();
  const Eigen::Vector3i srcDim = srcPart.getExtents();
  // Position in gather array
  const Eigen::Vector3i dstPos = srcPos + srcPart.getMin() - globalMin;
  const Eigen::Vector3i dstDim = globalMax - globalMin;
  if (numVoxels == 1) {
    // Read a single voxel
    T* srcGpuPtr = gpu_ptr(srcPart, srcQ, srcPos.x(), srcPos.y(), srcPos.z());
    T* dstGpuPtr =
        dst->gpu_ptr(dstPart, dstQ, dstPos.x(), dstPos.y(), dstPos.z());
    CUDA_RT_CALL(cudaMemcpyAsync(dstGpuPtr, srcGpuPtr, sizeof(T),
                                 cudaMemcpyDefault, stream));

  } else if (numVoxels > 1) {
    // Read a 3D volume
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
  }
}

template <class T>
void DistributionArray<T>::gatherSlice(Eigen::Vector3i slicePos,
                                       int srcQ,
                                       int dstQ,
                                       Partition srcPart,
                                       DistributionArray<T>* dst,
                                       cudaStream_t stream) {
  Eigen::Vector3i offset = slicePos - srcPart.getMin();

  Partition dstPart = dst->getAllocatedPartitions().at(0);
  Eigen::Vector3i srcLatDim = getExtents();
  Eigen::Vector3i dstLatDim = dst->getExtents();
  Eigen::Vector3i dstDim = dstPart.getArrayExtents();

  // Lattices must have same size
  if (srcLatDim != dstLatDim)
    throw std::out_of_range("Lattice sizes must be equal");

  // The destination partition must have the size of the entire lattice
  if (srcLatDim != dstDim)
    throw std::out_of_range(
        "Destination sub lattice must have size of entire lattice");

  // Copy the three planes which intersect at slicePos
  if (slicePos.x() >= srcPart.getMin().x() &&
      slicePos.x() < srcPart.getMax().x()) {
    // Offset source position to exclude ghostLayers from copy
    Eigen::Vector3i srcPos = srcPart.getGhostLayer();
    srcPos.x() += offset.x();
    // The destination is the global position of the source partition
    Eigen::Vector3i dstPos = srcPart.getMin();
    dstPos.x() = slicePos.x();
    // Dimensions of source parition must include ghostLayers
    Eigen::Vector3i srcDim = srcPart.getArrayExtents();
    // Copy the full extent of the source partition, excluding ghostLayers
    Eigen::Vector3i cpyExt = srcPart.getExtents();
    cpyExt.x() = 1;
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
  }
  if (slicePos.y() >= srcPart.getMin().y() &&
      slicePos.y() < srcPart.getMax().y()) {
    Eigen::Vector3i srcPos = srcPart.getGhostLayer();
    srcPos.y() += offset.y();
    Eigen::Vector3i dstPos = srcPart.getMin();
    dstPos.y() = slicePos.y();
    Eigen::Vector3i srcDim = srcPart.getArrayExtents();
    Eigen::Vector3i cpyExt = srcPart.getExtents();
    cpyExt.y() = 1;
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
  }
  if (slicePos.z() >= srcPart.getMin().z() &&
      slicePos.z() < srcPart.getMax().z()) {
    Eigen::Vector3i srcPos = srcPart.getGhostLayer();
    srcPos.z() += offset.z();
    Eigen::Vector3i dstPos = srcPart.getMin();
    dstPos.z() = slicePos.z();
    Eigen::Vector3i srcDim = srcPart.getArrayExtents();
    Eigen::Vector3i cpyExt = srcPart.getExtents();
    cpyExt.z() = 1;
    memcpy3DAsync(*this, srcPart, srcQ, srcPos, srcDim, dst, dstPart, dstQ,
                  dstPos, dstDim, cpyExt, stream);
  }
}

// Upload the distributions functions from the CPU to the GPU
template <class T>
DistributionArray<T>& DistributionArray<T>::upload() {
  for (std::pair<Partition, MemoryStore*> element : m_arrays)
    *element.second->gpu = *element.second->cpu;
  return *this;
}

// Download the distributions functions from the GPU to the CPU
template <class T>
DistributionArray<T>& DistributionArray<T>::download() {
  for (std::pair<Partition, MemoryStore*> element : m_arrays)
    *element.second->cpu = *element.second->gpu;
  return *this;
}

template <class T>
DistributionArray<T>& DistributionArray<T>::operator=(
    const DistributionArray<T>& f) {
  if (getExtents() == f.getExtents()) {
    for (std::pair<Partition, MemoryStore*> element : m_arrays) {
      Partition partition = element.first;
      MemoryStore* v1 = element.second;
      if (f.m_arrays.find(partition) != f.m_arrays.end()) {
        MemoryStore* v2 = f.m_arrays.at(partition);
        // thrust::copy(v2.gpu->begin(), v2.gpu->end(), v1.gpu->begin());
        thrust::copy(v2->cpu->begin(), v2->cpu->end(), v1->cpu->begin());
      } else {
        throw std::out_of_range(
            "RHS must have allocated all partitions of LHS");
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
    for (std::pair<Partition, MemoryStore*> element : f1->m_arrays) {
      Partition partition = element.first;
      MemoryStore* v1 = element.second;
      if (f2->m_arrays.find(partition) != f2->m_arrays.end()) {
        MemoryStore* v2 = f2->m_arrays.at(partition);
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
