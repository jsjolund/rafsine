// Generated by lbm_gen_d3q19f_bgk.py
#pragma once
#include "CudaUtils.hpp"
#include "PhysicalQuantity.hpp"
__device__ __forceinline__ void computeBGK(int x,
                                           int y,
                                           int z,
                                           int nx,
                                           int ny,
                                           int nz,
                                           real_t nu,
                                           real_t nuT,
                                           real_t C,
                                           real_t Pr_t,
                                           real_t gBetta,
                                           real_t Tref,
                                           real_t f0,
                                           real_t f1,
                                           real_t f2,
                                           real_t f3,
                                           real_t f4,
                                           real_t f5,
                                           real_t f6,
                                           real_t f7,
                                           real_t f8,
                                           real_t f9,
                                           real_t f10,
                                           real_t f11,
                                           real_t f12,
                                           real_t f13,
                                           real_t f14,
                                           real_t f15,
                                           real_t f16,
                                           real_t f17,
                                           real_t f18,
                                           real_t T0,
                                           real_t T1,
                                           real_t T2,
                                           real_t T3,
                                           real_t T4,
                                           real_t T5,
                                           real_t T6,
                                           real_t* __restrict__ df_tmp,
                                           real_t* __restrict__ dfT_tmp,
                                           PhysicalQuantity* phy) {
  real_t f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq, f9eq, f10eq,
      f11eq, f12eq, f13eq, f14eq, f15eq, f16eq, f17eq, f18eq;
  real_t f1neq, f2neq, f3neq, f4neq, f5neq, f6neq, f7neq, f8neq, f9neq, f10neq,
      f11neq, f12neq, f13neq, f14neq, f15neq, f16neq, f17neq, f18neq;
  real_t T0eq, T1eq, T2eq, T3eq, T4eq, T5eq, T6eq;
  real_t S_bar;
  real_t ST;
  real_t rho;
  real_t sq_term;
  real_t T;
  real_t Sxx, Syy, Szz, Sxy, Syz, Sxz;
  real_t tau_V, tau_T;
  real_t Fup, Fdown;
  real_t vx, vy, vz;

  // Macroscopic density
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
        f3 + f4 + f5 + f6 + f7 + f8 + f9;

  // Macroscopic velocity
  vx = (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) / rho;
  vy = (-f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 + f8 - f9) / rho;
  vz = (f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 + f5 - f6) / rho;

  // Macroscopic temperature
  T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
  sq_term = -1.5f * (vx) * (vx)-1.5f * (vy) * (vy)-1.5f * (vz) * (vz);

  // Compute the equilibrium distribution function
  f0eq = 0.33333333333333331f * rho * (sq_term + 1.0f);
  f1eq = 0.055555555555555552f * rho *
         (sq_term + 4.5f * (vx) * (vx) + 3.0f * vx + 1.0f);
  f2eq = 0.055555555555555552f * rho *
         (sq_term + 4.5f * (vx) * (vx)-3.0f * vx + 1.0f);
  f3eq = 0.055555555555555552f * rho *
         (sq_term + 4.5f * (vy) * (vy) + 3.0f * vy + 1.0f);
  f4eq = 0.055555555555555552f * rho *
         (sq_term + 4.5f * (vy) * (vy)-3.0f * vy + 1.0f);
  f5eq = 0.055555555555555552f * rho *
         (sq_term + 4.5f * (vz) * (vz) + 3.0f * vz + 1.0f);
  f6eq = 0.055555555555555552f * rho *
         (sq_term + 4.5f * (vz) * (vz)-3.0f * vz + 1.0f);
  f7eq =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx + 3.0f * vy + 4.5f * (vx + vy) * (vx + vy) + 1.0f);
  f8eq =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx + 3.0f * vy + 4.5f * (-vx + vy) * (-vx + vy) + 1.0f);
  f9eq =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx - 3.0f * vy + 4.5f * (vx - vy) * (vx - vy) + 1.0f);
  f10eq =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx - 3.0f * vy + 4.5f * (-vx - vy) * (-vx - vy) + 1.0f);
  f11eq =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx + 3.0f * vz + 4.5f * (vx + vz) * (vx + vz) + 1.0f);
  f12eq =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx + 3.0f * vz + 4.5f * (-vx + vz) * (-vx + vz) + 1.0f);
  f13eq =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx - 3.0f * vz + 4.5f * (vx - vz) * (vx - vz) + 1.0f);
  f14eq =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx - 3.0f * vz + 4.5f * (-vx - vz) * (-vx - vz) + 1.0f);
  f15eq =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vy + 3.0f * vz + 4.5f * (vy + vz) * (vy + vz) + 1.0f);
  f16eq =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vy + 3.0f * vz + 4.5f * (-vy + vz) * (-vy + vz) + 1.0f);
  f17eq =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vy - 3.0f * vz + 4.5f * (vy - vz) * (vy - vz) + 1.0f);
  f18eq =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vy - 3.0f * vz + 4.5f * (-vy - vz) * (-vy - vz) + 1.0f);

  // Temperature equilibirum distribution functions
  T0eq = 0.14285714285714285f * T;
  T1eq = 0.14285714285714285f * T * (3.5f * vx + 1.0f);
  T2eq = 0.14285714285714285f * T * (1.0f - 3.5f * vx);
  T3eq = 0.14285714285714285f * T * (3.5f * vy + 1.0f);
  T4eq = 0.14285714285714285f * T * (1.0f - 3.5f * vy);
  T5eq = 0.14285714285714285f * T * (3.5f * vz + 1.0f);
  T6eq = 0.14285714285714285f * T * (1.0f - 3.5f * vz);

  // Boussinesq approximation of body force
  Fup = gBetta * (T - Tref);
  Fdown = -gBetta * (T - Tref);

  // Difference to velocity equilibrium
  f1neq = f1 - f1eq;
  f2neq = f2 - f2eq;
  f3neq = f3 - f3eq;
  f4neq = f4 - f4eq;
  f5neq = f5 - f5eq;
  f6neq = f6 - f6eq;
  f7neq = f7 - f7eq;
  f8neq = f8 - f8eq;
  f9neq = f9 - f9eq;
  f10neq = f10 - f10eq;
  f11neq = f11 - f11eq;
  f12neq = f12 - f12eq;
  f13neq = f13 - f13eq;
  f14neq = f14 - f14eq;
  f15neq = f15 - f15eq;
  f16neq = f16 - f16eq;
  f17neq = f17 - f17eq;
  f18neq = f18 - f18eq;

  // Non equilibrium stress-tensor for velocity
  Sxx = f10neq + f11neq + f12neq + f13neq + f14neq + f1neq + f2neq + f7neq +
        f8neq + f9neq;
  Syy = f10neq + f15neq + f16neq + f17neq + f18neq + f3neq + f4neq + f7neq +
        f8neq + f9neq;
  Szz = f11neq + f12neq + f13neq + f14neq + f15neq + f16neq + f17neq + f18neq +
        f5neq + f6neq;
  Sxy = f10neq + f7neq - f8neq - f9neq;
  Sxz = f11neq - f12neq - f13neq + f14neq;
  Syz = f15neq - f16neq - f17neq + f18neq;

  // Magnitude of strain rate tensor
  S_bar = 1.4142135623730951f *
          powf(0.5f * (Sxx) * (Sxx) + (Sxy) * (Sxy) + (Sxz) * (Sxz) +
                   0.5f * (Syy) * (Syy) + (Syz) * (Syz) + 0.5f * (Szz) * (Szz),
               0.5f);
  ST = -0.16666666666666666f * nu +
       0.70710678118654746f *
           powf((C) * (C)*S_bar + 0.055555555555555552f * (nu) * (nu), 0.5f);

  // Modified relaxation time
  tau_V = 3.0f * ST + 3.0f * nu + 0.5f;

  // Modified relaxation time for the temperature
  tau_T = 3.0f * nuT + 0.5f + 3.0f * ST / Pr_t;

  // Relax velocity
  dftmp3D(0, x, y, z, nx, ny, nz) =
      f0 - 1.0f * f0 / tau_V + 1.0f * f0eq / tau_V;
  dftmp3D(1, x, y, z, nx, ny, nz) =
      f1 - 1.0f * f1 / tau_V + 1.0f * f1eq / tau_V;
  dftmp3D(2, x, y, z, nx, ny, nz) =
      f2 - 1.0f * f2 / tau_V + 1.0f * f2eq / tau_V;
  dftmp3D(3, x, y, z, nx, ny, nz) =
      f3 - 1.0f * f3 / tau_V + 1.0f * f3eq / tau_V;
  dftmp3D(4, x, y, z, nx, ny, nz) =
      f4 - 1.0f * f4 / tau_V + 1.0f * f4eq / tau_V;
  dftmp3D(5, x, y, z, nx, ny, nz) =
      Fup + f5 - 1.0f * f5 / tau_V + 1.0f * f5eq / tau_V;
  dftmp3D(6, x, y, z, nx, ny, nz) =
      Fdown + f6 - 1.0f * f6 / tau_V + 1.0f * f6eq / tau_V;
  dftmp3D(7, x, y, z, nx, ny, nz) =
      f7 - 1.0f * f7 / tau_V + 1.0f * f7eq / tau_V;
  dftmp3D(8, x, y, z, nx, ny, nz) =
      f8 - 1.0f * f8 / tau_V + 1.0f * f8eq / tau_V;
  dftmp3D(9, x, y, z, nx, ny, nz) =
      f9 - 1.0f * f9 / tau_V + 1.0f * f9eq / tau_V;
  dftmp3D(10, x, y, z, nx, ny, nz) =
      f10 - 1.0f * f10 / tau_V + 1.0f * f10eq / tau_V;
  dftmp3D(11, x, y, z, nx, ny, nz) =
      f11 - 1.0f * f11 / tau_V + 1.0f * f11eq / tau_V;
  dftmp3D(12, x, y, z, nx, ny, nz) =
      f12 - 1.0f * f12 / tau_V + 1.0f * f12eq / tau_V;
  dftmp3D(13, x, y, z, nx, ny, nz) =
      f13 - 1.0f * f13 / tau_V + 1.0f * f13eq / tau_V;
  dftmp3D(14, x, y, z, nx, ny, nz) =
      f14 - 1.0f * f14 / tau_V + 1.0f * f14eq / tau_V;
  dftmp3D(15, x, y, z, nx, ny, nz) =
      f15 - 1.0f * f15 / tau_V + 1.0f * f15eq / tau_V;
  dftmp3D(16, x, y, z, nx, ny, nz) =
      f16 - 1.0f * f16 / tau_V + 1.0f * f16eq / tau_V;
  dftmp3D(17, x, y, z, nx, ny, nz) =
      f17 - 1.0f * f17 / tau_V + 1.0f * f17eq / tau_V;
  dftmp3D(18, x, y, z, nx, ny, nz) =
      f18 - 1.0f * f18 / tau_V + 1.0f * f18eq / tau_V;

  // Relax temperature
  Tdftmp3D(0, x, y, z, nx, ny, nz) =
      T0 - 1.0f * T0 / tau_T + 1.0f * T0eq / tau_T;
  Tdftmp3D(1, x, y, z, nx, ny, nz) =
      T1 - 1.0f * T1 / tau_T + 1.0f * T1eq / tau_T;
  Tdftmp3D(2, x, y, z, nx, ny, nz) =
      T2 - 1.0f * T2 / tau_T + 1.0f * T2eq / tau_T;
  Tdftmp3D(3, x, y, z, nx, ny, nz) =
      T3 - 1.0f * T3 / tau_T + 1.0f * T3eq / tau_T;
  Tdftmp3D(4, x, y, z, nx, ny, nz) =
      T4 - 1.0f * T4 / tau_T + 1.0f * T4eq / tau_T;
  Tdftmp3D(5, x, y, z, nx, ny, nz) =
      T5 - 1.0f * T5 / tau_T + 1.0f * T5eq / tau_T;
  Tdftmp3D(6, x, y, z, nx, ny, nz) =
      T6 - 1.0f * T6 / tau_T + 1.0f * T6eq / tau_T;

  // Store macroscopic values
  phy->rho = rho;
  phy->T = T;
  phy->vx = vx;
  phy->vy = vy;
  phy->vz = vz;
}
