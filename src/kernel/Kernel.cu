#include "Kernel.hpp"

template <LBM::Enum METHOD, int QV, int QT, D3Q4::Enum AXIS>
__global__ void ComputeKernel(
    const Partition partition, real_t* __restrict__ df,
    real_t* __restrict__ df_tmp, real_t* __restrict__ dfT,
    real_t* __restrict__ dfT_tmp, real_t* __restrict__ dfTeff,
    real_t* __restrict__ dfTeff_tmp, const voxel_t* __restrict__ voxels,
    voxel_t* __restrict__ bcsId, VoxelType::Enum* __restrict__ bcsType,
    real_t* __restrict__ bcsTemperature, real3_t* __restrict__ bcsVelocity,
    int3* __restrict__ bcsNormal, int3* __restrict__ bcsRelPos,
    real_t* __restrict__ bcsTau1, real_t* __restrict__ bcsTau2,
    real_t* __restrict__ bcsLambda, const real_t dt, const real_t nu,
    const real_t C, const real_t nuT, const real_t Pr_t, const real_t gBetta,
    const real_t Tref, real_t* __restrict__ averageSrc,
    real_t* __restrict__ averageDst,
    const DisplayQuantity::Enum displayQuantity, real_t* __restrict__ plot) {
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

  /// STEP 1 STREAMING to [f0, f1, .. fQV].
  real_t fs[QV];
  real_t Ts[QT];
  streamV<QV>(x, y, z, nx, ny, nz, df, fs);
  streamT<QT>(x, y, z, nx, ny, nz, dfT, Ts);

  const real3_t v = make_float3(velocity.x, velocity.y, velocity.z);
  const real3_t n = make_float3(normal.x, normal.y, normal.z);

  const real_t* qVweights;
  switch (QV) {
    case 7:
      qVweights = D3Q7weights;
      break;
    case 19:
      qVweights = D3Q19weights;
      break;
    case 27:
      qVweights = D3Q27weights;
      break;
  }
  const real_t* qTweights;
  switch (QT) {
    case 7:
      qTweights = D3Q7weights;
      break;
    case 19:
      qTweights = D3Q19weights;
      break;
    case 27:
      qTweights = D3Q27weights;
      break;
  }

  if (type == VoxelType::WALL) {
    // Half-way bounceback

// BC for velocity dfs
#pragma unroll
    for (int i = 1; i < QV; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        fs[i] = df3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);
      }
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < QT; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        Ts[i] = Tdf3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);
      }
    }
    /////////////////////////////
  } else if (type == VoxelType::INLET_CONSTANT ||
             type == VoxelType::INLET_RELATIVE ||
             type == VoxelType::INLET_ZERO_GRADIENT) {
    // BC for velocity dfs
#pragma unroll
    for (int i = 1; i < QV; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      const real_t dot_vv = dot(v, v);
      if (dot(ei, n) > 0.0) {
        const real_t wi = qVweights[i];
        const real_t rho_0 = 1.0;
        const real_t dot_eiv = dot(ei, v);
        // If the velocity is zero, use half-way bounceback instead
        if (length(v) == 0.0) {
          fs[i] = df3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);

        } else {
          fs[i] =
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
        const real_t wi = qTweights[i];
        if (dot(ei, n) > 0.0) {
          Ts[i] = wi * temperature * (1.0 + 3.0 * dot(ei, v));
        }
      }
    } else if (type == VoxelType::INLET_ZERO_GRADIENT) {
#pragma unroll
      for (int i = 1; i < QT; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        if (dot(ei, n) > 0.0) {
          // Approximate a first order expansion
          Ts[i] =
              Tdf3D(i, x + normal.x, y + normal.y, z + normal.z, nx, ny, nz);
        }
      }
    } else if (type == VoxelType::INLET_RELATIVE) {
      // Compute macroscopic temperature at the relative position
      real_t Tamb = 0;
#pragma unroll
      for (int i = 1; i < QT; i++) {
        Tamb +=
            Tdf3D(i, x + rel_pos.x, y + rel_pos.y, z + rel_pos.z, nx, ny, nz);
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
        const real_t wi = qTweights[i];

        if (dot(ei, n) > 0.0) {
          Ts[i] = Tnew * wi * (1.0 + 3.0 * dot(ei, v));
        }
      }
      dfTeff_tmp[I4D(0, x, y, z, nx, ny, nz)] = Teff_new;
    }
  }

  PhysicalQuantity phy = {.rho = 0, .T = 0, .vx = 0, .vy = 0, .vz = 0};

  switch (METHOD) {
    case LBM::BGK:
      computeBGK(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, fs[0],
                 fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[7], fs[8], fs[9],
                 fs[10], fs[11], fs[12], fs[13], fs[14], fs[15], fs[16], fs[17],
                 fs[18], Ts[0], Ts[1], Ts[2], Ts[3], Ts[4], Ts[5], Ts[6],
                 df_tmp, dfT_tmp, &phy);
      break;

    case LBM::MRT:
      computeMRT(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, fs[0],
                 fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[7], fs[8], fs[9],
                 fs[10], fs[11], fs[12], fs[13], fs[14], fs[15], fs[16], fs[17],
                 fs[18], Ts[0], Ts[1], Ts[2], Ts[3], Ts[4], Ts[5], Ts[6],
                 df_tmp, dfT_tmp, &phy);
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
  X(LBM::MRT, 19, 7, D3Q4::ORIGIN) \
  X(LBM::MRT, 19, 7, D3Q4::X_AXIS) \
  X(LBM::MRT, 19, 7, D3Q4::Y_AXIS) \
  X(LBM::MRT, 19, 7, D3Q4::Z_AXIS)

#define X(METHOD, QV, QT, AXIS)                                                \
  template __global__ void ComputeKernel<METHOD, QV, QT, AXIS>(                \
      const Partition partition, real_t* __restrict__ df,                      \
      real_t* __restrict__ df_tmp, real_t* __restrict__ dfT,                   \
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
