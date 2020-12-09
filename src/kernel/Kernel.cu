#include "Kernel.hpp"

template <LBM::Enum METHOD, int QU, int QT, D3Q4::Enum AXIS>
__global__ void ComputeKernel(const Partition partition,
                              real_t* __restrict__ dfU,
                              real_t* __restrict__ dfU_tmp,
                              real_t* __restrict__ dfT,
                              real_t* __restrict__ dfT_tmp,
                              real_t* __restrict__ dfTeff,
                              real_t* __restrict__ dfTeff_tmp,
                              const voxel_t* __restrict__ voxels,
                              voxel_t* __restrict__ bcsId,
                              VoxelType::Enum* __restrict__ bcsType,
                              real_t* __restrict__ bcsTemperature,
                              real3_t* __restrict__ bcsVelocity,
                              int3* __restrict__ bcsNormal,
                              int3* __restrict__ bcsRelPos,
                              real_t* __restrict__ bcsTau1,
                              real_t* __restrict__ bcsTau2,
                              real_t* __restrict__ bcsLambda,
                              const real_t dt,
                              const real_t nu,
                              const real_t C,
                              const real_t nuT,
                              const real_t Pr_t,
                              const real_t gBetta,
                              const real_t Tref,
                              real_t* __restrict__ averageSrc,
                              real_t* __restrict__ averageDst,
                              const DisplayQuantity::Enum displayQuantity,
                              real_t* __restrict__ plot) {
  const Vector3<size_t> size = partition.getExtents();
  const Vector3<size_t> gl = partition.getGhostLayer();

  // Compute node position from thread indexes
  int tx, ty, tz;

  switch (AXIS) {
    case D3Q4::X_AXIS:
      tx = blockIdx.y * (size.x() - 1);  // Might not be multiple of 32
      ty = threadIdx.x;
      tz = blockIdx.x;
      break;

    case D3Q4::Y_AXIS:
      tx = threadIdx.x;
      ty = blockIdx.y * (size.y() - 1);
      tz = blockIdx.x;
      break;

    case D3Q4::Z_AXIS:
      tx = threadIdx.x;
      ty = blockIdx.x;
      tz = blockIdx.y * (size.z() - 1);
      break;

    case D3Q4::ORIGIN:
      tx = threadIdx.x + gl.x();
      ty = blockIdx.x + gl.y();
      tz = blockIdx.y + gl.z();
      break;
  }

  // Check that the thread is inside the simulation domain
  if ((tx >= size.x()) || (ty >= size.y()) || (tz >= size.z())) return;

  const Vector3<int> pos(tx, ty, tz);
  const voxel_t voxelID = voxels[I3D(pos, size)];

  // Plot empty voxels
  if (voxelID == VoxelType::Enum::EMPTY) {
    plot[I3D(pos, size)] = REAL_NAN;
    return;
  }

  const VoxelType::Enum type = bcsType[voxelID];
  const real_t temperature = bcsTemperature[voxelID];
  const real3_t velocity = bcsVelocity[voxelID];
  const int3 normal = bcsNormal[voxelID];
  const int3 rel_pos = bcsRelPos[voxelID];
  const real_t tau1 = bcsTau1[voxelID];
  const real_t tau2 = bcsTau2[voxelID];
  const real_t lambda = bcsLambda[voxelID];

  // Calculate array position for distribution functions (with ghostLayers)
  const int nx = size.x() + gl.x() * 2;
  const int ny = size.y() + gl.y() * 2;
  const int nz = size.z() + gl.z() * 2;

  const int x = pos.x() + gl.x();
  const int y = pos.y() + gl.y();
  const int z = pos.z() + gl.z();

  /// STEP 1 STREAMING to [f0, f1, .. fQU].
  real_t Ui[QU];
  real_t Ti[QT];
  stream<QU>(x, y, z, nx, ny, nz, dfU, Ui);
  stream<QT>(x, y, z, nx, ny, nz, dfT, Ti);

  const real3_t v = make_float3(velocity.x, velocity.y, velocity.z);
  const real3_t n = make_float3(normal.x, normal.y, normal.z);

  const real_t* Uwgt = getWeights<QU>();
  const real_t* Twgt = getWeights<QT>();

  if (type == VoxelType::WALL) {
    // Half-way bounceback

// BC for velocity dfs
#pragma unroll
    for (int i = 1; i < QU; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        Ui[i] = dfU[I4D(D3Q27Opposite[i], x, y, z, nx, ny, nz)];
      }
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < QT; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        Ti[i] = dfT[I4D(D3Q27Opposite[i], x, y, z, nx, ny, nz)];
      }
    }
    /////////////////////////////
  } else if (type == VoxelType::INLET_CONSTANT ||
             type == VoxelType::INLET_RELATIVE ||
             type == VoxelType::INLET_ZERO_GRADIENT) {
    // BC for velocity dfs
#pragma unroll
    for (int i = 1; i < QU; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      const real_t dot_vv = dot(v, v);
      if (dot(ei, n) > 0.0) {
        const real_t wi = Uwgt[i];
        const real_t rho_0 = 1.0;
        const real_t dot_eiv = dot(ei, v);
        // If the velocity is zero, use half-way bounceback instead
        if (length(v) == 0.0) {
          Ui[i] = dfU[I4D(D3Q27Opposite[i], x, y, z, nx, ny, nz)];
        } else {
          Ui[i] =
              wi * rho_0 *
              (1.0 + 3.0 * dot_eiv + 4.5 * dot_eiv * dot_eiv - 1.5 * dot_vv);
        }
      }
    }
    // BC for temperature dfs
    if (type == VoxelType::INLET_CONSTANT) {
#pragma unroll
      for (int i = 1; i < QT; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real_t wi = Twgt[i];
        if (dot(ei, n) > 0.0) {
          Ti[i] = wi * temperature * (1.0 + 3.0 * dot(ei, v));
        }
      }
    } else if (type == VoxelType::INLET_ZERO_GRADIENT) {
#pragma unroll
      for (int i = 1; i < QT; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        if (dot(ei, n) > 0.0) {
          // Approximate a first order expansion
          Ti[i] =
              dfT[I4D(i, x + normal.x, y + normal.y, z + normal.z, nx, ny, nz)];
        }
      }
    } else if (type == VoxelType::INLET_RELATIVE) {
      // Compute macroscopic temperature at the relative position
      real_t Tamb = 0;
#pragma unroll
      for (int i = 1; i < QT; i++) {
        Ti[i] = dfT[I4D(i, x + rel_pos.x, y + rel_pos.y, z + rel_pos.z, nx, ny,
                        nz)];
      }
      // Internal temperature
      const real_t Teff_old = dfTeff[I4D(0, x, y, z, nx, ny, nz)];

      const real_t Teff_new =
          tau1 / (tau1 + dt) * Teff_old +
          dt / (tau1 + dt) * (Tamb + (1.0 - lambda) * temperature);
      const real_t Tnew =
          Tamb + temperature +
          tau2 / (tau1 + dt) * (Teff_old - Tamb - (1.0 - lambda) * temperature);

#pragma unroll
      for (int i = 1; i < QT; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real_t wi = Twgt[i];

        if (dot(ei, n) > 0.0) { Ti[i] = Tnew * wi * (1.0 + 3.0 * dot(ei, v)); }
      }
      dfTeff_tmp[I4D(0, x, y, z, nx, ny, nz)] = Teff_new;
    }
  }

  PhysicalQuantity phy = {.rho = 0, .T = 0, .vx = 0, .vy = 0, .vz = 0};

  switch (METHOD) {
    case LBM::BGK:
      computeBGK(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, Ui[0],
                 Ui[1], Ui[2], Ui[3], Ui[4], Ui[5], Ui[6], Ui[7], Ui[8], Ui[9],
                 Ui[10], Ui[11], Ui[12], Ui[13], Ui[14], Ui[15], Ui[16], Ui[17],
                 Ui[18], Ti[0], Ti[1], Ti[2], Ti[3], Ti[4], Ti[5], Ti[6],
                 dfU_tmp, dfT_tmp, &phy);
      break;

    case LBM::MRT:
      computeMRT(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, Ui[0],
                 Ui[1], Ui[2], Ui[3], Ui[4], Ui[5], Ui[6], Ui[7], Ui[8], Ui[9],
                 Ui[10], Ui[11], Ui[12], Ui[13], Ui[14], Ui[15], Ui[16], Ui[17],
                 Ui[18], Ti[0], Ti[1], Ti[2], Ti[3], Ti[4], Ti[5], Ti[6],
                 dfU_tmp, dfT_tmp, &phy);
      break;

    case LBM::MRT27:
      computeMRT27(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, Ui[0],
                   Ui[1], Ui[2], Ui[3], Ui[4], Ui[5], Ui[6], Ui[7], Ui[8],
                   Ui[9], Ui[10], Ui[11], Ui[12], Ui[13], Ui[14], Ui[15],
                   Ui[16], Ui[17], Ui[18], Ui[19], Ui[20], Ui[21], Ui[22],
                   Ui[23], Ui[24], Ui[25], Ui[26], Ti[0], Ti[1], Ti[2], Ti[3],
                   Ti[4], Ti[5], Ti[6], dfU_tmp, dfT_tmp, &phy);
      break;
  }

  // Average temperature and velocity
  averageDst[I4D(0, pos, size)] = averageSrc[I4D(0, pos, size)] + phy.T;
  averageDst[I4D(1, pos, size)] = averageSrc[I4D(1, pos, size)] + phy.vx;
  averageDst[I4D(2, pos, size)] = averageSrc[I4D(2, pos, size)] + phy.vy;
  averageDst[I4D(3, pos, size)] = averageSrc[I4D(3, pos, size)] + phy.vz;

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

#define LBM_CONFIGS                \
  X(LBM::BGK, 19, 7, D3Q4::ORIGIN) \
  X(LBM::BGK, 19, 7, D3Q4::X_AXIS) \
  X(LBM::BGK, 19, 7, D3Q4::Y_AXIS) \
  X(LBM::BGK, 19, 7, D3Q4::Z_AXIS) \
  X(LBM::MRT27, 27, 7, D3Q4::ORIGIN) \
  X(LBM::MRT27, 27, 7, D3Q4::X_AXIS) \
  X(LBM::MRT27, 27, 7, D3Q4::Y_AXIS) \
  X(LBM::MRT27, 27, 7, D3Q4::Z_AXIS) \
  X(LBM::MRT, 19, 7, D3Q4::ORIGIN) \
  X(LBM::MRT, 19, 7, D3Q4::X_AXIS) \
  X(LBM::MRT, 19, 7, D3Q4::Y_AXIS) \
  X(LBM::MRT, 19, 7, D3Q4::Z_AXIS)

#define X(METHOD, QU, QT, AXIS)                                                \
  template __global__ void ComputeKernel<METHOD, QU, QT, AXIS>(                \
      const Partition partition, real_t* __restrict__ dfU,                     \
      real_t* __restrict__ dfU_tmp, real_t* __restrict__ dfT,                  \
      real_t* __restrict__ dfT_tmp, real_t* __restrict__ dfTeff,               \
      real_t* __restrict__ dfTeff_tmp, const voxel_t* __restrict__ voxels,     \
      voxel_t* __restrict__ bcsId, VoxelType::Enum* __restrict__ bcsType,      \
      real_t* __restrict__ bcsTemperature, real3_t* __restrict__ bcsVelocity,  \
      int3* __restrict__ bcsNormal, int3* __restrict__ bcsRelPos,              \
      real_t* __restrict__ bcsTau1, real_t* __restrict__ bcsTau2,              \
      real_t* __restrict__ bcsLambda, const real_t dt, const real_t nu,        \
      const real_t C, const real_t nuT, const real_t Pr_t,                     \
      const real_t gBetta, const real_t Tref, real_t* __restrict__ averageSrc, \
      real_t* __restrict__ averageDst,                                         \
      const DisplayQuantity::Enum displayQuantity, real_t* __restrict__ plot);
LBM_CONFIGS
#undef X
