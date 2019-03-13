#include "Kernel.hpp"

__global__ void InitKernel(real *__restrict__ df, real *__restrict__ dfT,
                           int nx, int ny, int nz, float rho, float vx,
                           float vy, float vz, float T, float sq_term) {
  glm::ivec3 pos(threadIdx.x, blockIdx.x, blockIdx.y);
  if ((pos.x >= nx) || (pos.y >= ny) || (pos.z >= nz)) return;
  const int x = pos.x;
  const int y = pos.y;
  const int z = pos.z;
  df3D(0, x, y, z, nx, ny, nz) = rho * (1.f / 3.f) * (1 + sq_term);
  df3D(1, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 + 3.f * vx + 4.5f * vx * vx + sq_term);
  df3D(2, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 - 3.f * vx + 4.5f * vx * vx + sq_term);
  df3D(3, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 + 3.f * vy + 4.5f * vy * vy + sq_term);
  df3D(4, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 - 3.f * vy + 4.5f * vy * vy + sq_term);
  df3D(5, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 + 3.f * vz + 4.5f * vz * vz + sq_term);
  df3D(6, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 - 3.f * vz + 4.5f * vz * vz + sq_term);
  df3D(7, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  df3D(8, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  df3D(9, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  df3D(10, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  df3D(11, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  df3D(12, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  df3D(13, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  df3D(14, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  df3D(15, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  df3D(16, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  df3D(17, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
  df3D(18, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

  Tdf3D(0, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1);
  Tdf3D(1, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
  Tdf3D(2, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
  Tdf3D(3, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
  Tdf3D(4, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
  Tdf3D(5, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
  Tdf3D(6, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);
}

__device__ void compute(
    // Partition
    const int ax, const int ay, const int az, const int anx, const int any,
    const int anz, const int hx, const int hy, const int hz,
    // Velocity distribution functions
    real *__restrict__ df, real *__restrict__ df_tmp,
    // Temperature distribution functions
    real *__restrict__ dfT, real *__restrict__ dfT_tmp,
    // Plot array for display
    real *__restrict__ plot,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    // Voxel type array
    const int *__restrict__ voxels,
    // Boundary condition data
    BoundaryCondition *__restrict__ bcs,
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
    // Quantity to be visualised
    const DisplayQuantity::Enum vis_q) {
  real f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
      f16, f17, f18;
  real f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq, f9eq, f10eq, f11eq,
      f12eq, f13eq, f14eq, f15eq, f16eq, f17eq, f18eq;
  real f1diff, f2diff, f3diff, f4diff, f5diff, f6diff, f7diff, f8diff, f9diff,
      f10diff, f11diff, f12diff, f13diff, f14diff, f15diff, f16diff, f17diff,
      f18diff;
  real T0, T1, T2, T3, T4, T5, T6;
  real T0eq, T1eq, T2eq, T3eq, T4eq, T5eq, T6eq;

  // Calculate array position for distribution functions (with halos)
  const int nx = anx + hx * 2;
  const int ny = any + hy * 2;
  const int nz = anz + hz * 2;

  const int x = ax + hx;
  const int y = ay + hy;
  const int z = az + hz;

  // Type of voxel for calculating boundary conditions
  voxel voxelID = voxels[I3D(ax, ay, az, anx, any, anz)];

  // Plot empty voxels
  if (voxelID == -1) {
    switch (vis_q) {
      case DisplayQuantity::VELOCITY_NORM:
        plot[I3D(ax, ay, az, anx, any, anz)] = REAL_NAN;
        break;
      case DisplayQuantity::DENSITY:
        plot[I3D(ax, ay, az, anx, any, anz)] = REAL_NAN;
        break;
      case DisplayQuantity::TEMPERATURE:
        plot[I3D(ax, ay, az, anx, any, anz)] = REAL_NAN;
        break;
    }
    return;
  }

  /// STEP 1 STREAMING
  // Store streamed distribution functions in registers
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

  f0 = df3D(0, x, y, z, nx, ny, nz);
  f1 = df3D(1, xm, y, z, nx, ny, nz);
  f2 = df3D(2, xp, y, z, nx, ny, nz);
  f3 = df3D(3, x, ym, z, nx, ny, nz);
  f4 = df3D(4, x, yp, z, nx, ny, nz);
  f5 = df3D(5, x, y, zm, nx, ny, nz);
  f6 = df3D(6, x, y, zp, nx, ny, nz);
  f7 = df3D(7, xm, ym, z, nx, ny, nz);
  f8 = df3D(8, xp, yp, z, nx, ny, nz);
  f9 = df3D(9, xm, yp, z, nx, ny, nz);
  f10 = df3D(10, xp, ym, z, nx, ny, nz);
  f11 = df3D(11, xm, y, zm, nx, ny, nz);
  f12 = df3D(12, xp, y, zp, nx, ny, nz);
  f13 = df3D(13, xm, y, zp, nx, ny, nz);
  f14 = df3D(14, xp, y, zm, nx, ny, nz);
  f15 = df3D(15, x, ym, zm, nx, ny, nz);
  f16 = df3D(16, x, yp, zp, nx, ny, nz);
  f17 = df3D(17, x, ym, zp, nx, ny, nz);
  f18 = df3D(18, x, yp, zm, nx, ny, nz);

  T0 = Tdf3D(0, x, y, z, nx, ny, nz);
  T1 = Tdf3D(1, xm, y, z, nx, ny, nz);
  T2 = Tdf3D(2, xp, y, z, nx, ny, nz);
  T3 = Tdf3D(3, x, ym, z, nx, ny, nz);
  T4 = Tdf3D(4, x, yp, z, nx, ny, nz);
  T5 = Tdf3D(5, x, y, zm, nx, ny, nz);
  T6 = Tdf3D(6, x, y, zp, nx, ny, nz);

  real *fs[19] = {&f0,  &f1,  &f2,  &f3,  &f4,  &f5,  &f6,  &f7,  &f8, &f9,
                  &f10, &f11, &f12, &f13, &f14, &f15, &f16, &f17, &f18};
  real *Ts[7] = {&T0, &T1, &T2, &T3, &T4, &T5, &T6};

  const BoundaryCondition bc = bcs[voxelID];

  if (bc.m_type == VoxelType::WALL) {
    // Generate inlet boundary condition
    real3 v = make_float3(bc.m_velocity.x, bc.m_velocity.y, bc.m_velocity.z);
    real3 n = make_float3(bc.m_normal.x, bc.m_normal.y, bc.m_normal.z);
// BC for velocity dfs
#pragma unroll
    for (int i = 0; i < 19; i++) {
      real3 ei = make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                             D3Q27directions[i * 3 + 2]);
      if (dot(ei, n) > 0.0)
        *fs[i] = df3D(D3Q27directionsOpposite[i], x, y, z, nx, ny, nz);
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < 7; i++) {
      real3 ei = make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                             D3Q27directions[i * 3 + 2]);
      if (dot(ei, n) > 0.0)
        *Ts[i] = Tdf3D(D3Q27directionsOpposite[i], x, y, z, nx, ny, nz);
    }
  } else if (bc.m_type == VoxelType::INLET_CONSTANT ||
             bc.m_type == VoxelType::INLET_RELATIVE ||
             bc.m_type == VoxelType::INLET_ZERO_GRADIENT) {
    // Generate inlet boundary condition
    real3 v = make_float3(bc.m_velocity.x, bc.m_velocity.y, bc.m_velocity.z);
    real3 n = make_float3(bc.m_normal.x, bc.m_normal.y, bc.m_normal.z);
// BC for velocity dfs
#pragma unroll
    for (int i = 0; i < 19; i++) {
      real3 ei = make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                             D3Q27directions[i * 3 + 2]);
      real dot_vv = dot(v, v);
      if (dot(ei, n) > 0.0) {
        real wi = D3Q19weights[i];
        real rho = 1.0;
        real dot_eiv = dot(ei, v);
        // if the velocity is zero, use half-way bounceback instead
        if (length(v) == 0.0) {
          *fs[i] = df3D(D3Q27directionsOpposite[i], x, y, z, nx, ny, nz);
        } else {
          *fs[i] = real(
              wi * rho *
              (1.0 + 3.0 * dot_eiv + 4.5 * dot_eiv * dot_eiv - 1.5 * dot_vv));
        }
      }
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < 7; i++) {
      real3 ei = make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                             D3Q27directions[i * 3 + 2]);
      real wi = D3Q7weights[i];
      if (dot(ei, n) > 0.0) {
        if (bc.m_type == VoxelType::INLET_CONSTANT) {
          *Ts[i] = real(wi * bc.m_temperature * (1.0 + 3.0 * dot(ei, v)));
        } else if (bc.m_type == VoxelType::INLET_ZERO_GRADIENT) {
          // approximate a first order expansion
          *Ts[i] = Tdf3D(i, x + bc.m_normal.x, y + bc.m_normal.y,
                         z + bc.m_normal.z, nx, ny, nz);
        } else if (bc.m_type == VoxelType::INLET_RELATIVE) {
          // compute macroscopic temperature at the relative position
          real Trel = 0;
#pragma unroll
          for (int qIdx = 1; qIdx < 7; qIdx++)
            Trel = Trel + Tdf3D(qIdx, x + bc.m_rel_pos.x, y + bc.m_rel_pos.y,
                                z + bc.m_rel_pos.z, nx, ny, nz);
          *Ts[i] =
              real((Trel + bc.m_temperature) * (wi * (1.0 + 3.0 * dot(ei, v))));
        }
      }
    }
  }

  // Compute physical quantities
  real rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 +
             f13 + f14 + f15 + f16 + f17 + f18;
  real T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
  real vx = (1 / rho) * (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14);
  real vy = (1 / rho) * (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18);
  real vz =
      (1 / rho) * (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18);

  // Average temperature and velocity
  averageDst[I4D(0, ax, ay, az, anx, any, anz)] =
      averageSrc[I4D(0, ax, ay, az, anx, any, anz)] + T;
  averageDst[I4D(1, ax, ay, az, anx, any, anz)] =
      averageSrc[I4D(1, ax, ay, az, anx, any, anz)] + vx;
  averageDst[I4D(2, ax, ay, az, anx, any, anz)] =
      averageSrc[I4D(2, ax, ay, az, anx, any, anz)] + vy;
  averageDst[I4D(3, ax, ay, az, anx, any, anz)] =
      averageSrc[I4D(3, ax, ay, az, anx, any, anz)] + vz;

  switch (vis_q) {
    case DisplayQuantity::VELOCITY_NORM:
      plot[I3D(ax, ay, az, anx, any, anz)] = sqrt(vx * vx + vy * vy + vz * vz);
      break;
    case DisplayQuantity::DENSITY:
      plot[I3D(ax, ay, az, anx, any, anz)] = rho;
      break;
    case DisplayQuantity::TEMPERATURE:
      plot[I3D(ax, ay, az, anx, any, anz)] = T;
      break;
  }

  // Compute the equilibrium distribution function
  real sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  f0eq = rho * (1.f / 3.f) * (1 + sq_term);
  f1eq = rho * (1.f / 18.f) * (1 + 3 * vx + 4.5f * vx * vx + sq_term);
  f2eq = rho * (1.f / 18.f) * (1 - 3 * vx + 4.5f * vx * vx + sq_term);
  f3eq = rho * (1.f / 18.f) * (1 + 3 * vy + 4.5f * vy * vy + sq_term);
  f4eq = rho * (1.f / 18.f) * (1 - 3 * vy + 4.5f * vy * vy + sq_term);
  f5eq = rho * (1.f / 18.f) * (1 + 3 * vz + 4.5f * vz * vz + sq_term);
  f6eq = rho * (1.f / 18.f) * (1 - 3 * vz + 4.5f * vz * vz + sq_term);
  f7eq = rho * (1.f / 36.f) *
         (1 + 3 * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  f8eq = rho * (1.f / 36.f) *
         (1 - 3 * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  f9eq = rho * (1.f / 36.f) *
         (1 + 3 * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  f10eq = rho * (1.f / 36.f) *
          (1 - 3 * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  f11eq = rho * (1.f / 36.f) *
          (1 + 3 * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  f12eq = rho * (1.f / 36.f) *
          (1 - 3 * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  f13eq = rho * (1.f / 36.f) *
          (1 + 3 * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  f14eq = rho * (1.f / 36.f) *
          (1 - 3 * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  f15eq = rho * (1.f / 36.f) *
          (1 + 3 * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  f16eq = rho * (1.f / 36.f) *
          (1 - 3 * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  f17eq = rho * (1.f / 36.f) *
          (1 + 3 * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
  f18eq = rho * (1.f / 36.f) *
          (1 - 3 * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

  // Compute the equilibrium temperature distribution
  T0eq = T * (1.f / 7.f);
  T1eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
  T2eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
  T3eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
  T4eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
  T5eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
  T6eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);

  // Difference to equilibrium
  f1diff = f1 - f1eq;
  f2diff = f2 - f2eq;
  f3diff = f3 - f3eq;
  f4diff = f4 - f4eq;
  f5diff = f5 - f5eq;
  f6diff = f6 - f6eq;
  f7diff = f7 - f7eq;
  f8diff = f8 - f8eq;
  f9diff = f9 - f9eq;
  f10diff = f10 - f10eq;
  f11diff = f11 - f11eq;
  f12diff = f12 - f12eq;
  f13diff = f13 - f13eq;
  f14diff = f14 - f14eq;
  f15diff = f15 - f15eq;
  f16diff = f16 - f16eq;
  f17diff = f17 - f17eq;
  f18diff = f18 - f18eq;

  // Non equilibrium stress-tensor for velocity
  real Pi_x_x = f1diff + f2diff + f7diff + f8diff + f9diff + f10diff + f11diff +
                f12diff + f13diff + f14diff;
  real Pi_x_y = f7diff + f8diff - f9diff - f10diff;
  real Pi_x_z = f11diff + f12diff - f13diff - f14diff;
  real Pi_y_y = f3diff + f4diff + f7diff + f8diff + f9diff + f10diff + f15diff +
                f16diff + f17diff + f18diff;
  real Pi_y_z = f15diff + f16diff - f17diff - f18diff;
  real Pi_z_z = f5diff + f6diff + f11diff + f12diff + f13diff + f14diff +
                f15diff + f16diff + f17diff + f18diff;
  // Variance
  real Q = Pi_x_x * Pi_x_x + 2 * Pi_x_y * Pi_x_y + 2 * Pi_x_z * Pi_x_z +
           Pi_y_y * Pi_y_y + 2 * Pi_y_z * Pi_y_z + Pi_z_z * Pi_z_z;
  // Local stress tensor
  real ST = (1 / (real)6) * (sqrt(nu * nu + 18 * C * C * sqrt(Q)) - nu);
  // Modified relaxation time
  real tau = 3 * (nu + ST) + (real)0.5;

  dftmp3D(0, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f0 + (1 / tau) * f0eq;
  dftmp3D(1, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f1 + (1 / tau) * f1eq;
  dftmp3D(2, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f2 + (1 / tau) * f2eq;
  dftmp3D(3, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f3 + (1 / tau) * f3eq;
  dftmp3D(4, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f4 + (1 / tau) * f4eq;
  dftmp3D(5, x, y, z, nx, ny, nz) =
      (1 - 1 / tau) * f5 + (1 / tau) * f5eq + 0.5f * gBetta * (T - Tref);
  dftmp3D(6, x, y, z, nx, ny, nz) =
      (1 - 1 / tau) * f6 + (1 / tau) * f6eq - 0.5f * gBetta * (T - Tref);
  dftmp3D(7, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f7 + (1 / tau) * f7eq;
  dftmp3D(8, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f8 + (1 / tau) * f8eq;
  dftmp3D(9, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f9 + (1 / tau) * f9eq;
  dftmp3D(10, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f10 + (1 / tau) * f10eq;
  dftmp3D(11, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f11 + (1 / tau) * f11eq;
  dftmp3D(12, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f12 + (1 / tau) * f12eq;
  dftmp3D(13, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f13 + (1 / tau) * f13eq;
  dftmp3D(14, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f14 + (1 / tau) * f14eq;
  dftmp3D(15, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f15 + (1 / tau) * f15eq;
  dftmp3D(16, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f16 + (1 / tau) * f16eq;
  dftmp3D(17, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f17 + (1 / tau) * f17eq;
  dftmp3D(18, x, y, z, nx, ny, nz) = (1 - 1 / tau) * f18 + (1 / tau) * f18eq;

  // Modified relaxation time for the temperature
  tau = 3 * (nuT + ST / Pr_t) + (real)0.5;
  // Relax temperature
  Tdftmp3D(0, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T0 + (1 / tau) * T0eq;
  Tdftmp3D(1, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T1 + (1 / tau) * T1eq;
  Tdftmp3D(2, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T2 + (1 / tau) * T2eq;
  Tdftmp3D(3, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T3 + (1 / tau) * T3eq;
  Tdftmp3D(4, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T4 + (1 / tau) * T4eq;
  Tdftmp3D(5, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T5 + (1 / tau) * T5eq;
  Tdftmp3D(6, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T6 + (1 / tau) * T6eq;
}

__global__ void ComputeKernel(
    // Partition
    const SubLattice subLattice,
    // Velocity distribution functions
    real *__restrict__ df, real *__restrict__ df_tmp,
    // Temperature distribution functions
    real *__restrict__ dfT, real *__restrict__ dfT_tmp,
    // Plot array for display
    real *__restrict__ plot,
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real *__restrict__ averageSrc, real *__restrict__ averageDst,
    // Voxel type array
    const int *__restrict__ voxels,
    // Boundary condition data
    BoundaryCondition *__restrict__ bcs,
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
    // Quantity to be visualised
    const DisplayQuantity::Enum vis_q) {
  // Compute node position from thread indexes
  glm::ivec3 threadPos(threadIdx.x, blockIdx.x, blockIdx.y);
  glm::ivec3 partSize = subLattice.getDims();
  glm::ivec3 partHalo = subLattice.getHalo();

  // Check that the thread is inside the simulation domain
  if ((threadPos.x >= partSize.x) || (threadPos.y >= partSize.y) ||
      (threadPos.z >= partSize.z))
    return;

  // Calculate array position and size for averaging (without halos)
  const int anx = partSize.x;
  const int any = partSize.y;
  const int anz = partSize.z;

  const int ax = threadPos.x;
  const int ay = threadPos.y;
  const int az = threadPos.z;

  const int hx = partHalo.x;
  const int hy = partHalo.y;
  const int hz = partHalo.z;

  compute(ax, ay, az, anx, any, anz, hx, hy, hz, df, df_tmp, dfT, dfT_tmp, plot,
          averageSrc, averageDst, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
          vis_q);
}
