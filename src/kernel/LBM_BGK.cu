#include "LBM-BGK.h"
__device__ void computeBGK(int x,
                           int y,
                           int z,
                           int nx,
                           int ny,
                           int nz,
                           real nu,
                           real nuT,
                           real C,
                           real Pr_t,
                           real gBetta,
                           real Tref,
                           real f0,
                           real f1,
                           real f2,
                           real f3,
                           real f4,
                           real f5,
                           real f6,
                           real f7,
                           real f8,
                           real f9,
                           real f10,
                           real f11,
                           real f12,
                           real f13,
                           real f14,
                           real f15,
                           real f16,
                           real f17,
                           real f18,
                           real T0,
                           real T1,
                           real T2,
                           real T3,
                           real T4,
                           real T5,
                           real T6,
                           real* __restrict__ df_tmp,
                           real* __restrict__ dfT_tmp,
                           PhysicalQuantity* phy) {
  real f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq, f9eq, f10eq, f11eq,
      f12eq, f13eq, f14eq, f15eq, f16eq, f17eq, f18eq;
  real f0neq, f1neq, f2neq, f3neq, f4neq, f5neq, f6neq, f7neq, f8neq, f9neq,
      f10neq, f11neq, f12neq, f13neq, f14neq, f15neq, f16neq, f17neq, f18neq;
  real T0eq, T1eq, T2eq, T3eq, T4eq, T5eq, T6eq;
  real S_bar;
  real ST;
  real rho;
  real sq_term;
  real T;
  real Sxx, Syy, Szz, Sxy, Syz, Sxz;
  real tau_V, tau_T;
  real Fup, Fdown;
  real vx, vy, vz;

  // Macroscopic density
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
        f3 + f4 + f5 + f6 + f7 + f8 + f9;

  // Macroscopic velocity
  vx = (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) / rho;
  vy = (f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9) / rho;
  vz = (f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6) / rho;

  // Macroscopic temperature
  T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
  sq_term = -1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2);

  // Compute the equilibrium distribution function
  f0eq = 0.33333333333333331 * rho * (sq_term + 1.0);
  f1eq = 0.055555555555555552 * rho *
         (sq_term + 4.5 * pow(vx, 2) + 3.0 * vx + 1.0);
  f2eq = 0.055555555555555552 * rho *
         (sq_term + 4.5 * pow(vx, 2) - 3.0 * vx + 1.0);
  f3eq = 0.055555555555555552 * rho *
         (sq_term + 4.5 * pow(vy, 2) + 3.0 * vy + 1.0);
  f4eq = 0.055555555555555552 * rho *
         (sq_term + 4.5 * pow(vy, 2) - 3.0 * vy + 1.0);
  f5eq = 0.055555555555555552 * rho *
         (sq_term + 4.5 * pow(vz, 2) + 3.0 * vz + 1.0);
  f6eq = 0.055555555555555552 * rho *
         (sq_term + 4.5 * pow(vz, 2) - 3.0 * vz + 1.0);
  f7eq = 0.027777777777777776 * rho *
         (sq_term + 3.0 * vx + 3.0 * vy + 4.5 * pow(vx + vy, 2) + 1.0);
  f8eq = 0.027777777777777776 * rho *
         (sq_term - 3.0 * vx - 3.0 * vy + 4.5 * pow(-vx - vy, 2) + 1.0);
  f9eq = 0.027777777777777776 * rho *
         (sq_term + 3.0 * vx - 3.0 * vy + 4.5 * pow(vx - vy, 2) + 1.0);
  f10eq = 0.027777777777777776 * rho *
          (sq_term - 3.0 * vx + 3.0 * vy + 4.5 * pow(-vx + vy, 2) + 1.0);
  f11eq = 0.027777777777777776 * rho *
          (sq_term + 3.0 * vx + 3.0 * vz + 4.5 * pow(vx + vz, 2) + 1.0);
  f12eq = 0.027777777777777776 * rho *
          (sq_term - 3.0 * vx - 3.0 * vz + 4.5 * pow(-vx - vz, 2) + 1.0);
  f13eq = 0.027777777777777776 * rho *
          (sq_term + 3.0 * vx - 3.0 * vz + 4.5 * pow(vx - vz, 2) + 1.0);
  f14eq = 0.027777777777777776 * rho *
          (sq_term - 3.0 * vx + 3.0 * vz + 4.5 * pow(-vx + vz, 2) + 1.0);
  f15eq = 0.027777777777777776 * rho *
          (sq_term + 3.0 * vy + 3.0 * vz + 4.5 * pow(vy + vz, 2) + 1.0);
  f16eq = 0.027777777777777776 * rho *
          (sq_term - 3.0 * vy - 3.0 * vz + 4.5 * pow(-vy - vz, 2) + 1.0);
  f17eq = 0.027777777777777776 * rho *
          (sq_term + 3.0 * vy - 3.0 * vz + 4.5 * pow(vy - vz, 2) + 1.0);
  f18eq = 0.027777777777777776 * rho *
          (sq_term - 3.0 * vy + 3.0 * vz + 4.5 * pow(-vy + vz, 2) + 1.0);

  // Temperature equilibirum distribution functions
  T0eq = 0.14285714285714285 * T;
  T1eq = 0.14285714285714285 * T * (3.5 * vx + 1.0);
  T2eq = 0.14285714285714285 * T * (1.0 - 3.5 * vx);
  T3eq = 0.14285714285714285 * T * (3.5 * vy + 1.0);
  T4eq = 0.14285714285714285 * T * (1.0 - 3.5 * vy);
  T5eq = 0.14285714285714285 * T * (3.5 * vz + 1.0);
  T6eq = 0.14285714285714285 * T * (1.0 - 3.5 * vz);

  // Boussinesq approximation of body force
  Fup = gBetta * (T - Tref);
  Fdown = -gBetta * (T - Tref);

  // Difference to velocity equilibrium
  f0neq = f0 - f0eq;
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
  Sxy = -f10neq + f7neq + f8neq - f9neq;
  Sxz = f11neq + f12neq - f13neq - f14neq;
  Syz = f15neq + f16neq - f17neq - f18neq;

  // Magnitude of strain rate tensor
  S_bar = sqrt(pow(Sxx, 2) + 2.0 * pow(Sxy, 2) + 2.0 * pow(Sxz, 2) +
               pow(Syy, 2) + 2.0 * pow(Syz, 2) + pow(Szz, 2));
  ST = -0.16666666666666666 * nu +
       0.16666666666666666 * sqrt(18.0 * pow(C, 2) * S_bar + pow(nu, 2));

  // Modified relaxation time
  tau_V = 3.0 * ST + 3.0 * nu + 0.5;

  // Modified relaxation time for the temperature
  tau_T = 3.0 * nuT + 0.5 + 3.0 * ST / Pr_t;

  // Relax velocity
  dftmp3D(0, x, y, z, nx, ny, nz) = f0 - 1.0 * f0 / tau_V + 1.0 * f0eq / tau_V;
  dftmp3D(1, x, y, z, nx, ny, nz) = f1 - 1.0 * f1 / tau_V + 1.0 * f1eq / tau_V;
  dftmp3D(2, x, y, z, nx, ny, nz) = f2 - 1.0 * f2 / tau_V + 1.0 * f2eq / tau_V;
  dftmp3D(3, x, y, z, nx, ny, nz) = f3 - 1.0 * f3 / tau_V + 1.0 * f3eq / tau_V;
  dftmp3D(4, x, y, z, nx, ny, nz) = f4 - 1.0 * f4 / tau_V + 1.0 * f4eq / tau_V;
  dftmp3D(5, x, y, z, nx, ny, nz) =
      Fup + f5 - 1.0 * f5 / tau_V + 1.0 * f5eq / tau_V;
  dftmp3D(6, x, y, z, nx, ny, nz) =
      Fdown + f6 - 1.0 * f6 / tau_V + 1.0 * f6eq / tau_V;
  dftmp3D(7, x, y, z, nx, ny, nz) = f7 - 1.0 * f7 / tau_V + 1.0 * f7eq / tau_V;
  dftmp3D(8, x, y, z, nx, ny, nz) = f8 - 1.0 * f8 / tau_V + 1.0 * f8eq / tau_V;
  dftmp3D(9, x, y, z, nx, ny, nz) = f9 - 1.0 * f9 / tau_V + 1.0 * f9eq / tau_V;
  dftmp3D(10, x, y, z, nx, ny, nz) =
      f10 - 1.0 * f10 / tau_V + 1.0 * f10eq / tau_V;
  dftmp3D(11, x, y, z, nx, ny, nz) =
      f11 - 1.0 * f11 / tau_V + 1.0 * f11eq / tau_V;
  dftmp3D(12, x, y, z, nx, ny, nz) =
      f12 - 1.0 * f12 / tau_V + 1.0 * f12eq / tau_V;
  dftmp3D(13, x, y, z, nx, ny, nz) =
      f13 - 1.0 * f13 / tau_V + 1.0 * f13eq / tau_V;
  dftmp3D(14, x, y, z, nx, ny, nz) =
      f14 - 1.0 * f14 / tau_V + 1.0 * f14eq / tau_V;
  dftmp3D(15, x, y, z, nx, ny, nz) =
      f15 - 1.0 * f15 / tau_V + 1.0 * f15eq / tau_V;
  dftmp3D(16, x, y, z, nx, ny, nz) =
      f16 - 1.0 * f16 / tau_V + 1.0 * f16eq / tau_V;
  dftmp3D(17, x, y, z, nx, ny, nz) =
      f17 - 1.0 * f17 / tau_V + 1.0 * f17eq / tau_V;
  dftmp3D(18, x, y, z, nx, ny, nz) =
      f18 - 1.0 * f18 / tau_V + 1.0 * f18eq / tau_V;

  // Relax temperature
  Tdftmp3D(0, x, y, z, nx, ny, nz) = T0 - 1.0 * T0 / tau_T + 1.0 * T0eq / tau_T;
  Tdftmp3D(1, x, y, z, nx, ny, nz) = T1 - 1.0 * T1 / tau_T + 1.0 * T1eq / tau_T;
  Tdftmp3D(2, x, y, z, nx, ny, nz) = T2 - 1.0 * T2 / tau_T + 1.0 * T2eq / tau_T;
  Tdftmp3D(3, x, y, z, nx, ny, nz) = T3 - 1.0 * T3 / tau_T + 1.0 * T3eq / tau_T;
  Tdftmp3D(4, x, y, z, nx, ny, nz) = T4 - 1.0 * T4 / tau_T + 1.0 * T4eq / tau_T;
  Tdftmp3D(5, x, y, z, nx, ny, nz) = T5 - 1.0 * T5 / tau_T + 1.0 * T5eq / tau_T;
  Tdftmp3D(6, x, y, z, nx, ny, nz) = T6 - 1.0 * T6 / tau_T + 1.0 * T6eq / tau_T;
  phy->rho = rho;
  phy->T = T;
  phy->vx = vx;
  phy->vy = vy;
  phy->vz = vz;
}