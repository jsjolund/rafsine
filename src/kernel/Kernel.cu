#include "Kernel.hpp"

template <LBM::Enum method, D3Q4::Enum axis>
__global__ void ComputeKernel(const Partition partition,
                              real* __restrict__ df,
                              real* __restrict__ df_tmp,
                              real* __restrict__ dfT,
                              real* __restrict__ dfT_tmp,
                              real* __restrict__ dfTeff,
                              real* __restrict__ dfTeff_tmp,
                              const voxel_t* __restrict__ voxels,
                              BoundaryCondition* __restrict__ bcs,
                              const real dt,
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
  const Eigen::Vector3i size = partition.getExtents();
  const Eigen::Vector3i gl = partition.getGhostLayer();

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

  Eigen::Vector3i pos(tx, ty, tz);
  const voxel_t voxelID = voxels[I3D(pos, size)];

  // Plot empty voxels
  if (voxelID == -1) {
  plot[I3D(pos, size)] = REAL_NAN;
    return;
  }

  const BoundaryCondition bc = bcs[voxelID];

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
          // Approximate a first order expansion
          *Ts[i] = Tdf3D(i, x + bc.m_normal.x(), y + bc.m_normal.y(),
                         z + bc.m_normal.z(), nx, ny, nz);
        }
      }
    } else if (bc.m_type == VoxelType::INLET_RELATIVE) {
      // Compute macroscopic temperature at the relative position
      real Tamb = 0;
#pragma unroll
      for (int i = 1; i < 7; i++) {
        Tamb += Tdf3D(i, x + bc.m_rel_pos.x(), y + bc.m_rel_pos.y(),
                      z + bc.m_rel_pos.z(), nx, ny, nz);
      }
      // Internal temperature
      real Teff_old = dfTeff[I4D(0, x, y, z, nx, ny, nz)];

      real Teff_new = bc.m_tau1 / (bc.m_tau1 + dt) * Teff_old +
                      dt / (bc.m_tau1 + dt) *
                          (Tamb + (1.0 - bc.m_lambda) * bc.m_temperature);
      real Tnew =
          Tamb + bc.m_temperature +
          bc.m_tau2 / (bc.m_tau1 + dt) *
              (Teff_old - Tamb - (1.0 - bc.m_lambda) * bc.m_temperature);

#pragma unroll
      for (int i = 1; i < 7; i++) {
        const real3 ei =
            make_float3(D3Q27[i * 3], D3Q27[i * 3 + 1], D3Q27[i * 3 + 2]);
        const real wi = D3Q7weights[i];

        if (dot(ei, n) > 0.0) { *Ts[i] = Tnew * wi * (1.0 + 3.0 * dot(ei, v)); }
      }
      dfTeff_tmp[I4D(0, x, y, z, nx, ny, nz)] = Teff_new;
    }
  }

  PhysicalQuantity phy = {.rho = 0, .T = 0, .vx = 0, .vy = 0, .vz = 0};

  // if (method == LBM::BGK) {
  //   computeBGK(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, f0, f1, f2,
  //              f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16,
  //              f17, f18, T0, T1, T2, T3, T4, T5, T6, df_tmp, dfT_tmp, &phy);
  // } else if (method == LBM::MRT) {
  //   computeMRT(x, y, z, nx, ny, nz, nu, nuT, C, Pr_t, gBetta, Tref, f0, f1, f2,
  //              f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16,
  //              f17, f18, T0, T1, T2, T3, T4, T5, T6, df_tmp, dfT_tmp, &phy);
  // }
  // Compute physical quantities
  const real rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 +
                   f12 + f13 + f14 + f15 + f16 + f17 + f18;
  const real T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
  const real vx =
      (1 / rho) * (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14);
  const real vy =
      (1 / rho) * (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18);
  const real vz =
      (1 / rho) * (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18);

  // Compute the equilibrium distribution function
  const real sq_term = -1.5f * (vx * vx + vy * vy + vz * vz);
  const real f0eq = rho * (1.f / 3.f) * (1 + sq_term);
  const real f1eq =
      rho * (1.f / 18.f) * (1 + 3 * vx + 4.5f * vx * vx + sq_term);
  const real f2eq =
      rho * (1.f / 18.f) * (1 - 3 * vx + 4.5f * vx * vx + sq_term);
  const real f3eq =
      rho * (1.f / 18.f) * (1 + 3 * vy + 4.5f * vy * vy + sq_term);
  const real f4eq =
      rho * (1.f / 18.f) * (1 - 3 * vy + 4.5f * vy * vy + sq_term);
  const real f5eq =
      rho * (1.f / 18.f) * (1 + 3 * vz + 4.5f * vz * vz + sq_term);
  const real f6eq =
      rho * (1.f / 18.f) * (1 - 3 * vz + 4.5f * vz * vz + sq_term);
  const real f7eq =
      rho * (1.f / 36.f) *
      (1 + 3 * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  const real f8eq =
      rho * (1.f / 36.f) *
      (1 - 3 * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  const real f9eq =
      rho * (1.f / 36.f) *
      (1 + 3 * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  const real f10eq =
      rho * (1.f / 36.f) *
      (1 - 3 * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  const real f11eq =
      rho * (1.f / 36.f) *
      (1 + 3 * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  const real f12eq =
      rho * (1.f / 36.f) *
      (1 - 3 * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  const real f13eq =
      rho * (1.f / 36.f) *
      (1 + 3 * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  const real f14eq =
      rho * (1.f / 36.f) *
      (1 - 3 * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  const real f15eq =
      rho * (1.f / 36.f) *
      (1 + 3 * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  const real f16eq =
      rho * (1.f / 36.f) *
      (1 - 3 * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  const real f17eq =
      rho * (1.f / 36.f) *
      (1 + 3 * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
  const real f18eq =
      rho * (1.f / 36.f) *
      (1 - 3 * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

  // Compute the equilibrium temperature distribution
  const real T0eq = T * (1.f / 7.f);
  const real T1eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
  const real T2eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
  const real T3eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
  const real T4eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
  const real T5eq = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
  const real T6eq = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);

  // Difference to equilibrium
  const real f1diff = f1 - f1eq;
  const real f2diff = f2 - f2eq;
  const real f3diff = f3 - f3eq;
  const real f4diff = f4 - f4eq;
  const real f5diff = f5 - f5eq;
  const real f6diff = f6 - f6eq;
  const real f7diff = f7 - f7eq;
  const real f8diff = f8 - f8eq;
  const real f9diff = f9 - f9eq;
  const real f10diff = f10 - f10eq;
  const real f11diff = f11 - f11eq;
  const real f12diff = f12 - f12eq;
  const real f13diff = f13 - f13eq;
  const real f14diff = f14 - f14eq;
  const real f15diff = f15 - f15eq;
  const real f16diff = f16 - f16eq;
  const real f17diff = f17 - f17eq;
  const real f18diff = f18 - f18eq;

  // Non equilibrium stress-tensor for velocity
  const real Pi_x_x = f1diff + f2diff + f7diff + f8diff + f9diff + f10diff +
                      f11diff + f12diff + f13diff + f14diff;
  const real Pi_x_y = f7diff + f8diff - f9diff - f10diff;
  const real Pi_x_z = f11diff + f12diff - f13diff - f14diff;
  const real Pi_y_y = f3diff + f4diff + f7diff + f8diff + f9diff + f10diff +
                      f15diff + f16diff + f17diff + f18diff;
  const real Pi_y_z = f15diff + f16diff - f17diff - f18diff;
  const real Pi_z_z = f5diff + f6diff + f11diff + f12diff + f13diff + f14diff +
                      f15diff + f16diff + f17diff + f18diff;
  // Variance
  const real Q = Pi_x_x * Pi_x_x + 2 * Pi_x_y * Pi_x_y + 2 * Pi_x_z * Pi_x_z +
                 Pi_y_y * Pi_y_y + 2 * Pi_y_z * Pi_y_z + Pi_z_z * Pi_z_z;
  // Local stress tensor
  const real ST = (1 / (real)6) * (sqrt(nu * nu + 18 * C * C * sqrt(Q)) - nu);
  // Modified relaxation time
  const real tau = 3 * (nu + ST) + (real)0.5;

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
  const real tauT = 3 * (nuT + ST / Pr_t) + (real)0.5;

  // Relax temperature
  Tdftmp3D(0, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T0 + (1 / tauT) * T0eq;
  Tdftmp3D(1, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T1 + (1 / tauT) * T1eq;
  Tdftmp3D(2, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T2 + (1 / tauT) * T2eq;
  Tdftmp3D(3, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T3 + (1 / tauT) * T3eq;
  Tdftmp3D(4, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T4 + (1 / tauT) * T4eq;
  Tdftmp3D(5, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T5 + (1 / tauT) * T5eq;
  Tdftmp3D(6, x, y, z, nx, ny, nz) = (1 - 1 / tauT) * T6 + (1 / tauT) * T6eq;

  phy.rho = rho;
  phy.T = T;
  phy.vx = vx;
  phy.vy = vy;
  phy.vz = vz;

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

#define X(METHOD, AXIS)                                                  \
  template __global__ void ComputeKernel<METHOD, AXIS>(                  \
      const Partition partition, real* __restrict__ df,                  \
      real* __restrict__ df_tmp, real* __restrict__ dfT,                 \
      real* __restrict__ dfT_tmp, real* __restrict__ dfTeff,             \
      real* __restrict__ dfTeff_tmp, const voxel_t* __restrict__ voxels, \
      BoundaryCondition* __restrict__ bcs, const real dt, const real nu, \
      const real C, const real nuT, const real Pr_t, const real gBetta,  \
      const real Tref, real* __restrict__ averageSrc,                    \
      real* __restrict__ averageDst,                                     \
      const DisplayQuantity::Enum displayQuantity, real* __restrict__ plot);
LBM_CONFIGS
#undef X
