#include "Kernel.hpp"

__device__ PhysicalQuantity compute(
    // Lattice position in partition
    const Eigen::Vector3i pos,
    // Size of partition
    const Eigen::Vector3i size,
    // Size of ghostLayer
    const Eigen::Vector3i ghostLayer,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
    // Viscosity
    const real nu,
    // Smagorinsky constant
    const real C,
    // Thermal diffusivity
    const real nuT,
    // Turbulent Prandtl number
    const real Pr_t,
    // Gravity times thermal expansion
    const real gBetta,
    // Reference temperature for Boussinesq
    const real Tref,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst) {
  const voxel_t voxelID = voxels[I3D(pos, size)];
  if (voxelID == -1) return {.rho = 0, .T = 0, .vx = 0, .vy = 0, .vz = 0};

  const BoundaryCondition bc = bcs[voxelID];

  // Calculate array position for distribution functions (with ghostLayers)
  const int nx = size.x() + ghostLayer.x() * 2;
  const int ny = size.y() + ghostLayer.y() * 2;
  const int nz = size.z() + ghostLayer.z() * 2;

  const int x = pos.x() + ghostLayer.x();
  const int y = pos.y() + ghostLayer.y();
  const int z = pos.z() + ghostLayer.z();

  /// STEP 1 STREAMING
  // Store streamed distribution functions in registers
  // Modulo with wraparound for negative numbers
  const int xp = ((x + 1) % nx + nx) % nx;
  // x minus 1
  const int xm = ((x - 1) % nx + nx) % nx;
  // y plus 1
  const int yp = ((y + 1) % ny + ny) % ny;
  // y minus 1
  const int ym = ((y - 1) % ny + ny) % ny;
  // z plus 1
  const int zp = ((z + 1) % nz + nz) % nz;
  // z minus 1
  const int zm = ((z - 1) % nz + nz) % nz;

  real f0 = df3D(0, x, y, z, nx, ny, nz);
  real f1 = df3D(1, xm, y, z, nx, ny, nz);
  real f2 = df3D(2, xp, y, z, nx, ny, nz);
  real f3 = df3D(3, x, ym, z, nx, ny, nz);
  real f4 = df3D(4, x, yp, z, nx, ny, nz);
  real f5 = df3D(5, x, y, zm, nx, ny, nz);
  real f6 = df3D(6, x, y, zp, nx, ny, nz);
  real f7 = df3D(7, xm, ym, z, nx, ny, nz);
  real f8 = df3D(8, xp, yp, z, nx, ny, nz);
  real f9 = df3D(9, xm, yp, z, nx, ny, nz);
  real f10 = df3D(10, xp, ym, z, nx, ny, nz);
  real f11 = df3D(11, xm, y, zm, nx, ny, nz);
  real f12 = df3D(12, xp, y, zp, nx, ny, nz);
  real f13 = df3D(13, xm, y, zp, nx, ny, nz);
  real f14 = df3D(14, xp, y, zm, nx, ny, nz);
  real f15 = df3D(15, x, ym, zm, nx, ny, nz);
  real f16 = df3D(16, x, yp, zp, nx, ny, nz);
  real f17 = df3D(17, x, ym, zp, nx, ny, nz);
  real f18 = df3D(18, x, yp, zm, nx, ny, nz);

  real T0 = Tdf3D(0, x, y, z, nx, ny, nz);
  real T1 = Tdf3D(1, xm, y, z, nx, ny, nz);
  real T2 = Tdf3D(2, xp, y, z, nx, ny, nz);
  real T3 = Tdf3D(3, x, ym, z, nx, ny, nz);
  real T4 = Tdf3D(4, x, yp, z, nx, ny, nz);
  real T5 = Tdf3D(5, x, y, zm, nx, ny, nz);
  real T6 = Tdf3D(6, x, y, zp, nx, ny, nz);

  real* fs[19] = {&f0,  &f1,  &f2,  &f3,  &f4,  &f5,  &f6,  &f7,  &f8, &f9,
                  &f10, &f11, &f12, &f13, &f14, &f15, &f16, &f17, &f18};
  real* Ts[7] = {&T0, &T1, &T2, &T3, &T4, &T5, &T6};

  const real3 v =
      make_float3(bc.m_velocity.x(), bc.m_velocity.y(), bc.m_velocity.z());
  const real3 n =
      make_float3(bc.m_normal.x(), bc.m_normal.y(), bc.m_normal.z());

  if (bc.m_type == VoxelType::WALL) {
    // Half-way bounceback

// BC for velocity dfs
#pragma unroll
    for (int i = 1; i < 19; i++) {
      const real3 ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        *fs[i] = df3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);
      }
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < 7; i++) {
      const real3 ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        *Ts[i] = Tdf3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);
      }
    }
    /////////////////////////////
  } else if (bc.m_type == VoxelType::INLET_CONSTANT ||
             bc.m_type == VoxelType::INLET_RELATIVE ||
             bc.m_type == VoxelType::INLET_ZERO_GRADIENT) {
// BC for velocity dfs
#pragma unroll
    for (int i = 1; i < 19; i++) {
      const real3 ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      const real dot_vv = dot(v, v);
      if (dot(ei, n) > 0.0) {
        const real wi = D3Q19weights[i];
        const real rho_0 = 1.0;
        const real dot_eiv = dot(ei, v);
        // if the velocity is zero, use half-way bounceback instead
        if (length(v) == 0.0) {
          *fs[i] = df3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);

        } else {
          *fs[i] =
              wi * rho_0 *
              (1.0 + 3.0 * dot_eiv + 4.5 * dot_eiv * dot_eiv - 1.5 * dot_vv);
        }
      }
    }
    // BC for temperature dfs
    if (bc.m_type == VoxelType::INLET_CONSTANT) {
#pragma unroll
      for (int i = 1; i < 7; i++) {
        const real3 ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real wi = D3Q7weights[i];
        if (dot(ei, n) > 0.0) {
          *Ts[i] = wi * bc.m_temperature * (1.0 + 3.0 * dot(ei, v));
        }
      }
    } else if (bc.m_type == VoxelType::INLET_ZERO_GRADIENT) {
#pragma unroll
      for (int i = 1; i < 7; i++) {
        const real3 ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        if (dot(ei, n) > 0.0) {
          // approximate a first order expansion
          *Ts[i] = Tdf3D(i, x + bc.m_normal.x(), y + bc.m_normal.y(),
                         z + bc.m_normal.z(), nx, ny, nz);
        }
      }
      //     } else if (bc.m_type == VoxelType::INLET_RELATIVE) {
      //       // compute macroscopic temperature at the relative position
      //       real Trel = 0;
      //       real Told = 0;
      // #pragma unroll
      //       for (int i = 1; i < 7; i++) {
      //         Trel += Tdf3D(i, x + bc.m_rel_pos.x(), y + bc.m_rel_pos.y(),
      //                       z + bc.m_rel_pos.z(), nx, ny, nz);
      //         Told += Tdf3D(i, x, y, z, nx, ny, nz);
      //       }
      //       real tau = 2;
      //       real dt = 1;
      //       real lambda = 0.001;
      //       real Tdelta = bc.m_temperature;
      //       real Tnew = tau / (tau + dt) * Told +
      //                   dt / (tau + dt) * (Trel + (1.0 - lambda) * Tdelta);
      // #pragma unroll
      //       for (int i = 1; i < 7; i++) {
      //         const real3 ei =
      //             make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 +
      //             2]);
      //         const real wi = D3Q7weights[i];

      //         if (dot(ei, n) > 0.0) { *Ts[i] = Tnew * wi * (1.0 + 3.0 *
      //         dot(ei, v)); }
      //       }
      //     }

    } else if (bc.m_type == VoxelType::INLET_RELATIVE) {
      // compute macroscopic temperature at the relative position
      real Trel = 0;
#pragma unroll
      for (int i = 1; i < 7; i++) {
        Trel += Tdf3D(i, x + bc.m_rel_pos.x(), y + bc.m_rel_pos.y(),
                      z + bc.m_rel_pos.z(), nx, ny, nz);
      }
#pragma unroll
      for (int i = 1; i < 7; i++) {
        const real3 ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real wi = D3Q7weights[i];
        if (dot(ei, n) > 0.0) {
          *Ts[i] = (Trel + bc.m_temperature) * (wi * (1.0 + 3.0 * dot(ei, v)));
        }
      }
    }

  }

#include "LBM-BGK.h"
  // #include "LBM-MRT.h"
  // #include "LBM-MRT-BGK.h"

  const PhysicalQuantity phy = {
      .rho = rho, .T = T, .vx = vx, .vy = vy, .vz = vz};

  // Average temperature and velocity
  averageDst[I4D(0, pos, size)] = averageSrc[I4D(0, pos, size)] + phy.T;
  averageDst[I4D(1, pos, size)] = averageSrc[I4D(1, pos, size)] + phy.vx;
  averageDst[I4D(2, pos, size)] = averageSrc[I4D(2, pos, size)] + phy.vy;
  averageDst[I4D(3, pos, size)] = averageSrc[I4D(3, pos, size)] + phy.vz;

  return phy;
}

__device__ void computeAndPlot(
    // Lattice position in partition
    const Eigen::Vector3i pos,
    // Size of partition
    const Eigen::Vector3i size,
    // Size of ghostLayer
    const Eigen::Vector3i ghostLayer,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
    // Viscosity
    const real nu,
    // Smagorinsky constant
    const real C,
    // Thermal diffusivity
    const real nuT,
    // Turbulent Prandtl number
    const real Pr_t,
    // Gravity times thermal expansion
    const real gBetta,
    // Reference temperature for Boussinesq
    const real Tref,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    // Quantity to be visualised
    const DisplayQuantity::Enum displayQuantity,
    // Plot array for display
    real* __restrict__ plot) {
  // Type of voxel for calculating boundary conditions
  const voxel_t voxelID = voxels[I3D(pos, size)];

  // Plot empty voxels
  if (voxelID == -1) {
    plot[I3D(pos, size)] = REAL_NAN;
    return;
  }

  PhysicalQuantity phy =
      compute(pos, size, ghostLayer, df, df_tmp, dfT, dfT_tmp, voxels, bcs, nu,
              C, nuT, Pr_t, gBetta, Tref, averageSrc, averageDst);

  switch (displayQuantity) {
    case DisplayQuantity::VELOCITY_NORM:
      plot[I3D(pos, size)] =
          sqrt(phy.vx * phy.vx + phy.vy * phy.vy + phy.vz * phy.vz);
      break;
    case DisplayQuantity::DENSITY:
      plot[I3D(pos, size)] = phy.rho;
      break;
    case DisplayQuantity::TEMPERATURE:
      plot[I3D(pos, size)] = phy.T;
      break;
  }
}

__global__ void ComputeKernelInterior(const Partition partition,
                                      real* __restrict__ df,
                                      real* __restrict__ df_tmp,
                                      real* __restrict__ dfT,
                                      real* __restrict__ dfT_tmp,
                                      const voxel_t* __restrict__ voxels,
                                      BoundaryCondition* __restrict__ bcs,
                                      const real nu,
                                      const real C,
                                      const real nuT,
                                      const real Pr_t,
                                      const real gBetta,
                                      const real Tref,
                                      real* __restrict__ averageSrc,
                                      real* __restrict__ averageDst) {
  Eigen::Vector3i partSize = partition.getExtents();
  Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x + partGhostLayer.x();
  const int y = blockIdx.x + partGhostLayer.y();
  const int z = blockIdx.y + partGhostLayer.z();

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeKernelBoundaryX(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = blockIdx.y * (partSize.x() - 1);  // Might not be multiple of 32
  const int y = threadIdx.x;
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeKernelBoundaryY(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.y * (partSize.y() - 1);
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeKernelBoundaryZ(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y * (partSize.z() - 1);

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeAndPlotKernelInterior(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot) {
  Eigen::Vector3i partSize = partition.getExtents();
  Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x + partGhostLayer.x();
  const int y = blockIdx.x + partGhostLayer.y();
  const int z = blockIdx.y + partGhostLayer.z();

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}

__global__ void ComputeAndPlotKernelBoundaryX(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = blockIdx.y * (partSize.x() - 1);  // Might not be multiple of 32
  const int y = threadIdx.x;
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}

__global__ void ComputeAndPlotKernelBoundaryY(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.y * (partSize.y() - 1);
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}

__global__ void ComputeAndPlotKernelBoundaryZ(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
    const real nu,
    const real C,
    const real nuT,
    const real Pr_t,
    const real gBetta,
    const real Tref,
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity,
    real* __restrict__ plot) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y * (partSize.z() - 1);

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}
