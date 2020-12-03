#include "KernelInterface.hpp"

void KernelInterface::calculateAverages() {
  thrust::host_vector<real_t>* avgs =
      m_avgs->getHostVector(m_avgs->getPartition());
  for (size_t srcDev = 0; srcDev < m_nd; srcDev++) {
    SimulationState* state = m_state.at(srcDev);
    thrust::host_vector<real_t> avgPartial =
        *state->avgResult->getHostVector(state->avgResult->getPartition());
    thrust::host_vector<int> avgStencil = *state->avgStencil;
    thrust::counting_iterator<int> iter(0);

    thrust::gather_if(iter, iter + avgPartial.size(), avgStencil.begin(),
                      avgPartial.begin(), avgs->begin());
  }
  // Averages must be uploaded to run transform_reduce on GPU in
  // KernelInterface::getAverage...
  m_avgs->upload();
}

LatticeAverage KernelInterface::getAverage(VoxelCuboid vol,
                                           uint64_t deltaTicks) {
  unsigned int offset = m_avgOffsets[vol];
  unsigned int size = vol.getNumVoxels();
  real_t ticks = static_cast<real_t>(deltaTicks);
  Partition partition = m_avgs->getPartition();
  real_t temperature = m_avgs->getAverage(partition, 0, offset, size, ticks);
  real_t velocityX = m_avgs->getAverage(partition, 1, offset, size, ticks);
  real_t velocityY = m_avgs->getAverage(partition, 2, offset, size, ticks);
  real_t velocityZ = m_avgs->getAverage(partition, 3, offset, size, ticks);
  return LatticeAverage(temperature, velocityX, velocityY, velocityZ);
}

void KernelInterface::uploadBCs(std::shared_ptr<BoundaryConditions> bcs) {
#pragma omp parallel num_threads(m_nd)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    SimulationState* state = m_state.at(srcDev);
    for (size_t i = 0; i < bcs->size(); i++) {
      (*state->bcs_id)[i] = bcs->at(i).m_id;
      (*state->bcs_type)[i] = bcs->at(i).m_type;
      (*state->bcs_temperature)[i] = bcs->at(i).m_temperature;
      (*state->bcs_velocity)[i] =
          make_float3(bcs->at(i).m_velocity.x(), bcs->at(i).m_velocity.y(),
                      bcs->at(i).m_velocity.z());
      (*state->bcs_normal)[i] =
          make_int3(bcs->at(i).m_normal.x(), bcs->at(i).m_normal.y(),
                    bcs->at(i).m_normal.z());
      (*state->bcs_rel_pos)[i] =
          make_int3(bcs->at(i).m_rel_pos.x(), bcs->at(i).m_rel_pos.y(),
                    bcs->at(i).m_rel_pos.z());
      (*state->bcs_tau1)[i] = bcs->at(i).m_tau1;
      (*state->bcs_tau2)[i] = bcs->at(i).m_tau2;
      (*state->bcs_lambda)[i] = bcs->at(i).m_lambda;
    }
  }
}

void KernelInterface::getMinMax(real_t* min,
                                real_t* max,
                                thrust::host_vector<real_t>* histogram) {
  // *min = 20.0f;
  // *max = 30.0f;
  *min = REAL_MAX;
  *max = REAL_MIN;
  thrust::host_vector<real_t> mins(m_nd);
  thrust::host_vector<real_t> maxes(m_nd);
  thrust::fill(histogram->begin(), histogram->end(), 0.0);

#pragma omp parallel num_threads(m_nd)
  {
    const int srcDev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(srcDev));
    const Partition partition = getDevicePartition(srcDev);
    const Partition partitionNoGhostLayer(
        partition.getMin(), partition.getMax(), Vector3<int>(0, 0, 0));
    SimulationState* state = m_state.at(srcDev);
    mins[srcDev] = state->plot->getMin(partitionNoGhostLayer);
    maxes[srcDev] = state->plot->getMax(partitionNoGhostLayer);
#pragma omp barrier
#pragma omp single
    {
      *max = *thrust::max_element(maxes.begin(), maxes.end());
      *min = *thrust::min_element(mins.begin(), mins.end());
    }
    int nBins = histogram->size();
    thrust::host_vector<int> result(nBins);
    LatticeHistogram lHist;
    thrust::device_vector<real_t>* input =
        state->plot->getDeviceVector(partitionNoGhostLayer);
    lHist.calculate(input, *min, *max, nBins, &result);
#pragma omp critical
    for (int i = 0; i < nBins; i++) (*histogram)[i] += result[i];
  }
  for (size_t i = 0; i < histogram->size(); i++) (*histogram)[i] /= getSize();
}

// template KernelInterface<LBM::Enum::BGK, 19, 7, D3Q4::Enum::Y_AXIS>(
//     const size_t nx,
//     const size_t ny,
//     const size_t nz,
//     const real_t dt,
//     const std::shared_ptr<SimulationParams> cmptParams,
//     const std::shared_ptr<BoundaryConditions> bcs,
//     const std::shared_ptr<VoxelArray> voxels,
//     const std::shared_ptr<VoxelCuboidArray> avgVols,
//     const size_t nd,
//     const LBM::Enum method,
//     const D3Q4::Enum partitioning);
