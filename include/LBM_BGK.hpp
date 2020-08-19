#pragma once
#include "CudaUtils.hpp"
#include "PhysicalQuantity.hpp"
__device__ __forceinline__ PhysicalQuantity
computeBGK(const int x,
           const int y,
           const int z,
           const int nx,
           const int ny,
           const int nz,
           const real nu,
           const real nuT,
           const real C,
           const real Pr_t,
           const real gBetta,
           const real Tref,
           const real f0,
           const real f1,
           const real f2,
           const real f3,
           const real f4,
           const real f5,
           const real f6,
           const real f7,
           const real f8,
           const real f9,
           const real f10,
           const real f11,
           const real f12,
           const real f13,
           const real f14,
           const real f15,
           const real f16,
           const real f17,
           const real f18,
           const real T0,
           const real T1,
           const real T2,
           const real T3,
           const real T4,
           const real T5,
           const real T6,
           real* __restrict__ df_tmp,
           real* __restrict__ dfT_tmp,
           PhysicalQuantity* phy) {
  real f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq, f9eq, f10eq, f11eq,
      f12eq, f13eq, f14eq, f15eq, f16eq, f17eq, f18eq;
  real f1diff, f2diff, f3diff, f4diff, f5diff, f6diff, f7diff, f8diff, f9diff,
      f10diff, f11diff, f12diff, f13diff, f14diff, f15diff, f16diff, f17diff,
      f18diff;
  real T0eq, T1eq, T2eq, T3eq, T4eq, T5eq, T6eq;

  // Compute physical quantities
  real rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 +
             f13 + f14 + f15 + f16 + f17 + f18;
  real T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
  real vx = (1 / rho) * (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14);
  real vy = (1 / rho) * (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18);
  real vz =
      (1 / rho) * (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18);

  // compute the equilibrium distribution function
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

  // compute the equilibrium temperature distribution
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

  // non equilibrium stress-tensor for velocity
  real Pi_x_x = f1diff + f2diff + f7diff + f8diff + f9diff + f10diff + f11diff +
                f12diff + f13diff + f14diff;
  real Pi_x_y = f7diff + f8diff - f9diff - f10diff;
  real Pi_x_z = f11diff + f12diff - f13diff - f14diff;
  real Pi_y_y = f3diff + f4diff + f7diff + f8diff + f9diff + f10diff + f15diff +
                f16diff + f17diff + f18diff;
  real Pi_y_z = f15diff + f16diff - f17diff - f18diff;
  real Pi_z_z = f5diff + f6diff + f11diff + f12diff + f13diff + f14diff +
                f15diff + f16diff + f17diff + f18diff;

  // variance
  real Q = Pi_x_x * Pi_x_x + 2 * Pi_x_y * Pi_x_y + 2 * Pi_x_z * Pi_x_z +
           Pi_y_y * Pi_y_y + 2 * Pi_y_z * Pi_y_z + Pi_z_z * Pi_z_z;

  // local stress tensor
  real ST = (1 / (real)6) * (sqrt(nu * nu + 18 * C * C * sqrt(Q)) - nu);

  // modified relaxation time
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

  // modified relaxation time for the temperature
  tau = 3 * (nuT + ST / Pr_t) + (real)0.5;

  // relax temperature
  Tdftmp3D(0, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T0 + (1 / tau) * T0eq;
  Tdftmp3D(1, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T1 + (1 / tau) * T1eq;
  Tdftmp3D(2, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T2 + (1 / tau) * T2eq;
  Tdftmp3D(3, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T3 + (1 / tau) * T3eq;
  Tdftmp3D(4, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T4 + (1 / tau) * T4eq;
  Tdftmp3D(5, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T5 + (1 / tau) * T5eq;
  Tdftmp3D(6, x, y, z, nx, ny, nz) = (1 - 1 / tau) * T6 + (1 / tau) * T6eq;

  phy->rho = rho;
  phy->T = T;
  phy->vx = vx;
  phy->vy = vy;
  phy->vz = vz;
  return *phy;
}
