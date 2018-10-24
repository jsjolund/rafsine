#include "KernelData.hpp"

void KernelData::initDomain(float rho, float vx, float vy, float vz, float T) {
  /// Initialise distribution functions on the CPU
  float sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
#pragma omp parallel for
  for (int i = 0; i < m_df->getLatticeDims().x; ++i)
    for (int j = 0; j < m_df->getLatticeDims().y; ++j)
      for (int k = 0; k < m_df->getLatticeDims().z; ++k) {
        (*m_df)(0, i, j, k) = rho * (1.f / 3.f) * (1 + sq_term);
        (*m_df)(1, i, j, k) =
            rho * (1.f / 18.f) * (1 + 3.f * vx + 4.5f * vx * vx + sq_term);
        (*m_df)(2, i, j, k) =
            rho * (1.f / 18.f) * (1 - 3.f * vx + 4.5f * vx * vx + sq_term);
        (*m_df)(3, i, j, k) =
            rho * (1.f / 18.f) * (1 + 3.f * vy + 4.5f * vy * vy + sq_term);
        (*m_df)(4, i, j, k) =
            rho * (1.f / 18.f) * (1 - 3.f * vy + 4.5f * vy * vy + sq_term);
        (*m_df)(5, i, j, k) =
            rho * (1.f / 18.f) * (1 + 3.f * vz + 4.5f * vz * vz + sq_term);
        (*m_df)(6, i, j, k) =
            rho * (1.f / 18.f) * (1 - 3.f * vz + 4.5f * vz * vz + sq_term);
        (*m_df)(7, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
        (*m_df)(8, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
        (*m_df)(9, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
        (*m_df)(10, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
        (*m_df)(11, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
        (*m_df)(12, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
        (*m_df)(13, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
        (*m_df)(14, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
        (*m_df)(15, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
        (*m_df)(16, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
        (*m_df)(17, i, j, k) =
            rho * (1.f / 36.f) *
            (1 + 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
        (*m_df)(18, i, j, k) =
            rho * (1.f / 36.f) *
            (1 - 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

        (*m_dfT)(0, i, j, k) = T * (1.f / 7.f) * (1);
        (*m_dfT)(1, i, j, k) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
        (*m_dfT)(2, i, j, k) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
        (*m_dfT)(3, i, j, k) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
        (*m_dfT)(4, i, j, k) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
        (*m_dfT)(5, i, j, k) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
        (*m_dfT)(6, i, j, k) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);
      }
  // Upload them to the GPU
  m_df->upload();
  m_dfT->upload();
  m_df_tmp = m_df;
  m_dfT_tmp = m_dfT;
  m_df_tmp->upload();
  m_dfT_tmp->upload();
}

void KernelData::compute(real *plotGpuPointer,
                         DisplayQuantity::Enum displayQuantity) {
  // CUDA threads organization
  ComputeKernel<<<*m_grid_size, *m_block_size>>>(
      m_df->gpu_ptr(), m_df_tmp->gpu_ptr(), m_dfT->gpu_ptr(),
      m_dfT_tmp->gpu_ptr(), plotGpuPointer, m_voxels->gpu_ptr(), m_params->nx,
      m_params->ny, m_params->nz, m_params->nu, m_params->C, m_params->nuT,
      m_params->Pr_t, m_params->gBetta, m_params->Tref, displayQuantity,
      m_average->gpu_ptr(), bcs_gpu_ptr());
  DistributionFunctionsGroup::swap(*m_df, *m_df_tmp);
  DistributionFunctionsGroup::swap(*m_dfT, *m_dfT_tmp);
}

KernelData::~KernelData() {
  delete m_df, m_dfT, m_df_tmp, m_dfT_tmp, m_average, m_grid_size, m_block_size,
      m_bcs_d;
}

KernelData::KernelData(KernelParameters *params, BoundaryConditionsArray *bcs,
                       VoxelArray *voxels)
    : m_params(params), m_voxels(voxels) {
  int nx = params->nx;
  int ny = params->ny;
  int nz = params->nz;

  std::cout << "Domain size : (" << nx << ", " << ny << ", " << nz << ")"
            << std::endl;
  std::cout << "Total number of nodes : " << nx * ny * nz << std::endl;

  std::cout << "Uploading voxels" << std::endl;
  m_voxels->upload();

  std::cout << "Allocating distribution functions" << std::endl;
  // Allocate memory for the velocity distribution functions
  m_df = new DistributionFunctionsGroup(19, nx, ny, nz);
  // Allocate memory for the temperature distribution functions
  m_dfT = new DistributionFunctionsGroup(7, nx, ny, nz);
  // Allocate memory for the temporary distribution functions
  m_df_tmp = new DistributionFunctionsGroup(19, nx, ny, nz);
  // Allocate memory for the temporary temperature distribution function
  m_dfT_tmp = new DistributionFunctionsGroup(7, nx, ny, nz);

  // Initialise the distribution functions
  std::cout << "Initializing domain" << std::endl;
  initDomain(1.0, 0, 0, 0, params->Tinit);

  // Data for averaging are stored in the same structure
  // 0 -> temperature
  // 1 -> x-component of velocity
  // 2 -> y-component of velocity
  // 3 -> z-component of velocity
  std::cout << "Allocating averages" << std::endl;
  m_average = new DistributionFunctionsGroup(4, nx, ny, nz);
  m_average->fill(0, 0);
  m_average->fill(1, 0);
  m_average->fill(2, 0);
  m_average->fill(3, 0);

  m_grid_size = new dim3(ny, nz, 1);
  m_block_size = new dim3(nx, 1, 1);

  std::cout << "Uploading boundary conditions" << std::endl;
  m_bcs_d = new thrust::device_vector<BoundaryCondition>(*bcs);
}

void KernelData::uploadBCs(BoundaryConditionsArray *bcs) { *m_bcs_d = *bcs; }

void KernelData::resetAverages() {
  m_average->fill(0, 0);
  m_average->fill(1, 0);
  m_average->fill(2, 0);
  m_average->fill(3, 0);
  m_average->upload();
}
