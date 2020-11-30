#include "Kernel.hpp"

template <LBM::Enum method, D3Q4::Enum axis>
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

  switch (axis) {
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

  // const BoundaryCondition bc = bcs[voxelID];
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

#include "DdQqIndexing.hpp"

  real_t* fs[19] = {&f0,  &f1,  &f2,  &f3,  &f4,  &f5,  &f6,  &f7,  &f8, &f9,
                    &f10, &f11, &f12, &f13, &f14, &f15, &f16, &f17, &f18};
  real_t* Ts[7] = {&T0, &T1, &T2, &T3, &T4, &T5, &T6};

  const real3_t v = make_float3(velocity.x, velocity.y, velocity.z);
  const real3_t n = make_float3(normal.x, normal.y, normal.z);

  if (type == VoxelType::WALL) {
    // Half-way bounceback

// BC for velocity dfs
#pragma unroll
    for (int i = 1; i < 19; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        *fs[i] = df3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);
      }
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < 7; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      if (dot(ei, n) > 0.0) {
        *Ts[i] = Tdf3D(D3Q27Opposite[i], x, y, z, nx, ny, nz);
      }
    }
    /////////////////////////////
  } else if (type == VoxelType::INLET_CONSTANT ||
             type == VoxelType::INLET_RELATIVE ||
             type == VoxelType::INLET_ZERO_GRADIENT) {
// BC for velocity dfs
#pragma unroll
    for (int i = 1; i < 19; i++) {
      const real3_t ei =
          make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
      const real_t dot_vv = dot(v, v);
      if (dot(ei, n) > 0.0) {
        const real_t wi = D3Q19weights[i];
        const real_t rho_0 = 1.0;
        const real_t dot_eiv = dot(ei, v);
        // If the velocity is zero, use half-way bounceback instead
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
    if (type == VoxelType::INLET_CONSTANT) {
#pragma unroll
      for (int i = 1; i < 7; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real_t wi = D3Q7weights[i];
        if (dot(ei, n) > 0.0) {
          *Ts[i] = wi * temperature * (1.0 + 3.0 * dot(ei, v));
        }
      }
    } else if (type == VoxelType::INLET_ZERO_GRADIENT) {
#pragma unroll
      for (int i = 1; i < 7; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        if (dot(ei, n) > 0.0) {
          // Approximate a first order expansion
          *Ts[i] =
              Tdf3D(i, x + normal.x, y + normal.y, z + normal.z, nx, ny, nz);
        }
      }
    } else if (type == VoxelType::INLET_RELATIVE) {
      // Compute macroscopic temperature at the relative position
      real_t Tamb = 0;
#pragma unroll
      for (int i = 1; i < 7; i++) {
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
      for (int i = 1; i < 7; i++) {
        const real3_t ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real_t wi = D3Q7weights[i];

        if (dot(ei, n) > 0.0) {
          *Ts[i] = Tnew * wi * (1.0 + 3.0 * dot(ei, v));
        }
      }
      dfTeff_tmp[I4D(0, x, y, z, nx, ny, nz)] = Teff_new;
    }
  }

  PhysicalQuantity phy = {.rho = 0, .T = 0, .vx = 0, .vy = 0, .vz = 0};

  switch (method) {
    case LBM::BGK:
      computeBGK(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, f0, f1,
                 f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
                 f16, f17, f18, T0, T1, T2, T3, T4, T5, T6, df_tmp, dfT_tmp,
                 &phy);
      break;

    case LBM::MRT:
      computeMRT(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, f0, f1,
                 f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
                 f16, f17, f18, T0, T1, T2, T3, T4, T5, T6, df_tmp, dfT_tmp,
                 &phy);
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

#define LBM_CONFIGS         \
  X(LBM::BGK, D3Q4::ORIGIN) \
  X(LBM::BGK, D3Q4::X_AXIS) \
  X(LBM::BGK, D3Q4::Y_AXIS) \
  X(LBM::BGK, D3Q4::Z_AXIS) \
  X(LBM::MRT, D3Q4::ORIGIN) \
  X(LBM::MRT, D3Q4::X_AXIS) \
  X(LBM::MRT, D3Q4::Y_AXIS) \
  X(LBM::MRT, D3Q4::Z_AXIS)

#define X(METHOD, AXIS)                                                        \
  template __global__ void ComputeKernel<METHOD, AXIS>(                        \
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
