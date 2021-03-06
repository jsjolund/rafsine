// Generated by lbm_gen_d3q19f_mrt.py
#pragma once
#include "CudaUtils.hpp"
#include "PhysicalQuantity.hpp"
__device__ __forceinline__ void computeMRT(int x,
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
  real_t m1eq, m2eq, m4eq, m6eq, m8eq, m9eq, m10eq, m11eq, m12eq, m13eq, m14eq,
      m15eq, m16eq, m17eq, m18eq;
  real_t m1diff, m2diff, m4diff, m6diff, m8diff, m9diff, m10diff, m11diff,
      m12diff, m13diff, m14diff, m15diff, m16diff, m17diff, m18diff;
  real_t omega0, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8,
      omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16,
      omega17, omega18;
  real_t n1eq, n2eq, n3eq, n4eq, n5eq, n6eq;
  real_t n1diff, n2diff, n3diff, n4diff, n5diff, n6diff;
  real_t omegaT0, omegaT1, omegaT2, omegaT3, omegaT4, omegaT5, omegaT6;
  real_t S_bar;
  real_t ST;
  real_t T;
  real_t Fup;
  real_t Fdown;
  real_t tau_V;
  real_t tau_T;
  real_t Sxx, Syy, Szz, Sxy, Syz, Sxz;
  real_t rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx3, pixx3, pww, piww, pxy,
      pyz, pxz, mx, my, mz;
  real_t omega_e, omega_xx, omega_ej;
  real_t vx, vy, vz;

  // Macroscopic density
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
        f3 + f4 + f5 + f6 + f7 + f8 + f9;

  // Part of kinetic energy independent of density
  en = -30.0f * f0 - 11.0f * f1 + 8.0f * f10 + 8.0f * f11 + 8.0f * f12 +
       8.0f * f13 + 8.0f * f14 + 8.0f * f15 + 8.0f * f16 + 8.0f * f17 +
       8.0f * f18 - 11.0f * f2 - 11.0f * f3 - 11.0f * f4 - 11.0f * f5 -
       11.0f * f6 + 8.0f * f7 + 8.0f * f8 + 8.0f * f9;

  // Kinetic energy square
  epsilon = 12.0f * f0 - 4.0f * f1 + 1.0f * f10 + 1.0f * f11 + 1.0f * f12 +
            1.0f * f13 + 1.0f * f14 + 1.0f * f15 + 1.0f * f16 + 1.0f * f17 +
            1.0f * f18 - 4.0f * f2 - 4.0f * f3 - 4.0f * f4 - 4.0f * f5 -
            4.0f * f6 + 1.0f * f7 + 1.0f * f8 + 1.0f * f9;

  // Momentum
  jx = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
  jy = -f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 + f8 - f9;
  jz = f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 + f5 - f6;

  // Energy flux independent of mass flux
  qx = -4.0f * f1 - 1.0f * f10 + 1.0f * f11 - 1.0f * f12 + 1.0f * f13 -
       1.0f * f14 + 4.0f * f2 + 1.0f * f7 - 1.0f * f8 + 1.0f * f9;
  qy = -1.0f * f10 + 1.0f * f15 - 1.0f * f16 + 1.0f * f17 - 1.0f * f18 -
       4.0f * f3 + 4.0f * f4 + 1.0f * f7 + 1.0f * f8 - 1.0f * f9;
  qz = 1.0f * f11 + 1.0f * f12 - 1.0f * f13 - 1.0f * f14 + 1.0f * f15 +
       1.0f * f16 - 1.0f * f17 - 1.0f * f18 - 4.0f * f5 + 4.0f * f6;

  // Symmetric traceless viscous stress tensor
  pxx3 = 2.0f * f1 + 1.0f * f10 + 1.0f * f11 + 1.0f * f12 + 1.0f * f13 +
         1.0f * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0f * f2 - f3 -
         f4 - f5 - f6 + 1.0f * f7 + 1.0f * f8 + 1.0f * f9;
  pww = f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9;
  pxy = f10 + f7 - f8 - f9;
  pyz = f15 - f16 - f17 + f18;
  pxz = f11 - f12 - f13 + f14;

  // Fourth order moments
  pixx3 = -4.0f * f1 + 1.0f * f10 + 1.0f * f11 + 1.0f * f12 + 1.0f * f13 +
          1.0f * f14 - 2.0f * f15 - 2.0f * f16 - 2.0f * f17 - 2.0f * f18 -
          4.0f * f2 + 2.0f * f3 + 2.0f * f4 + 2.0f * f5 + 2.0f * f6 +
          1.0f * f7 + 1.0f * f8 + 1.0f * f9;
  piww = 1.0f * f10 - 1.0f * f11 - 1.0f * f12 - 1.0f * f13 - 1.0f * f14 -
         2.0f * f3 - 2.0f * f4 + 2.0f * f5 + 2.0f * f6 + 1.0f * f7 + 1.0f * f8 +
         1.0f * f9;

  // Antisymmetric third-order moment
  mx = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9;
  my = f10 + f15 - f16 + f17 - f18 - f7 - f8 + f9;
  mz = f11 + f12 - f13 - f14 - f15 - f16 + f17 + f18;

  // Model stability constants
  omega_e = 0;
  omega_xx = 0;
  omega_ej = -7.5396825396825395f;

  // Macroscopic velocity
  vx = jx / rho;
  vy = jy / rho;
  vz = jz / rho;

  // Velocity moment equilibirum distribution functions
  m1eq = 19.0f * (jx) * (jx) + 19.0f * (jy) * (jy) +
         19.0f * (jz) * (jz)-11.0f * rho;
  m2eq = omega_e * rho +
         1.0f * omega_ej * ((jx) * (jx) + (jy) * (jy) + (jz) * (jz));
  m4eq = -0.66666666666666663f * jx;
  m6eq = -0.66666666666666663f * jy;
  m8eq = -0.66666666666666663f * jz;
  m9eq = 0.66666666666666663f * (jx) * (jx)-0.33333333333333331f * (jy) *
         (jy)-0.33333333333333331f * (jz) * (jz);
  m10eq = omega_xx * (0.66666666666666663f * (jx) * (jx)-0.33333333333333331f *
                      (jy) * (jy)-0.33333333333333331f * (jz) * (jz));
  m11eq = 1.0f * (jy) * (jy)-1.0f * (jz) * (jz);
  m12eq = omega_xx * (1.0f * (jy) * (jy)-1.0f * (jz) * (jz));
  m13eq = 1.0f * jx * jy;
  m14eq = 1.0f * jy * jz;
  m15eq = 1.0f * jx * jz;
  m16eq = 0;
  m17eq = 0;
  m18eq = 0;

  // Difference to velocity equilibrium
  m1diff = en - m1eq;
  m2diff = epsilon - m2eq;
  m4diff = -m4eq + qx;
  m6diff = -m6eq + qy;
  m8diff = -m8eq + qz;
  m9diff = -m9eq + pxx3;
  m10diff = -m10eq + pixx3;
  m11diff = -m11eq + pww;
  m12diff = -m12eq + piww;
  m13diff = -m13eq + pxy;
  m14diff = -m14eq + pyz;
  m15diff = -m15eq + pxz;
  m16diff = -m16eq + mx;
  m17diff = -m17eq + my;
  m18diff = -m18eq + mz;

  // LES strain rate tensor
  Sxx = 0.93947368421052613f * f0 + 0.34447368421052627f * f1 -
        0.25052631578947365f * f10 - 0.25052631578947365f * f11 -
        0.25052631578947365f * f12 - 0.25052631578947365f * f13 -
        0.25052631578947365f * f14 - 0.25052631578947365f * f15 -
        0.25052631578947365f * f16 - 0.25052631578947365f * f17 -
        0.25052631578947365f * f18 + 0.34447368421052627f * f2 +
        0.34447368421052627f * f3 + 0.34447368421052627f * f4 +
        0.34447368421052627f * f5 + 0.34447368421052627f * f6 -
        0.25052631578947365f * f7 - 0.25052631578947365f * f8 -
        0.25052631578947365f * f9 -
        0.5f *
            (2.0f * f1 + 1.0f * f10 + 1.0f * f11 + 1.0f * f12 + 1.0f * f13 +
             1.0f * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0f * f2 -
             f3 - f4 - f5 - f6 + 1.0f * f7 + 1.0f * f8 + 1.0f * f9) /
            (3.0f * nu + 0.5f);
  Syy = 0.93947368421052613f * f0 + 0.34447368421052627f * f1 -
        0.25052631578947365f * f10 - 0.25052631578947365f * f11 -
        0.25052631578947365f * f12 - 0.25052631578947365f * f13 -
        0.25052631578947365f * f14 - 0.25052631578947365f * f15 -
        0.25052631578947365f * f16 - 0.25052631578947365f * f17 -
        0.25052631578947365f * f18 + 0.34447368421052627f * f2 +
        0.34447368421052627f * f3 + 0.34447368421052627f * f4 +
        0.34447368421052627f * f5 + 0.34447368421052627f * f6 -
        0.25052631578947365f * f7 - 0.25052631578947365f * f8 -
        0.25052631578947365f * f9 +
        0.25f *
            (2.0f * f1 - 2.0f * f10 + 4.0f * f11 + 4.0f * f12 + 4.0f * f13 +
             4.0f * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0f * f2 -
             4.0f * f3 - 4.0f * f4 + 2.0f * f5 + 2.0f * f6 - 2.0f * f7 -
             2.0f * f8 - 2.0f * f9) /
            (3.0f * nu + 0.5f);
  Szz = 0.93947368421052613f * f0 + 0.34447368421052627f * f1 -
        0.25052631578947365f * f10 - 0.25052631578947365f * f11 -
        0.25052631578947365f * f12 - 0.25052631578947365f * f13 -
        0.25052631578947365f * f14 - 0.25052631578947365f * f15 -
        0.25052631578947365f * f16 - 0.25052631578947365f * f17 -
        0.25052631578947365f * f18 + 0.34447368421052627f * f2 +
        0.34447368421052627f * f3 + 0.34447368421052627f * f4 +
        0.34447368421052627f * f5 + 0.34447368421052627f * f6 -
        0.25052631578947365f * f7 - 0.25052631578947365f * f8 -
        0.25052631578947365f * f9 +
        0.25f *
            (2.0f * f1 + 4.0f * f10 - 2.0f * f11 - 2.0f * f12 - 2.0f * f13 -
             2.0f * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0f * f2 +
             2.0f * f3 + 2.0f * f4 - 4.0f * f5 - 4.0f * f6 + 4.0f * f7 +
             4.0f * f8 + 4.0f * f9) /
            (3.0f * nu + 0.5f);
  Sxy = -1.5f * (f10 + f7 - f8 - f9) / (3.0f * nu + 0.5f);
  Syz = -1.5f * (f15 - f16 - f17 + f18) / (3.0f * nu + 0.5f);
  Sxz = -1.5f * (f11 - f12 - f13 + f14) / (3.0f * nu + 0.5f);

  // Magnitude of strain rate tensor
  S_bar = 2.0f *
          powf(0.5f * (Sxx) * (Sxx) + (Sxy) * (Sxy) + (Sxz) * (Sxz) +
                   0.5f * (Syy) * (Syy) + (Syz) * (Syz) + 0.5f * (Szz) * (Szz),
               0.5f);

  // Filtered strain rate
  ST = 4.0f * (C) * (C)*S_bar;

  // Macroscopic temperature
  T = T0 + T1 + T2 + T3 + T4 + T5 + T6;

  // Temperature moment equilibirum distribution functions
  n1eq = T * vx;
  n2eq = T * vy;
  n3eq = T * vz;
  n4eq = 0.75f * T;
  n5eq = 0.0f;
  n6eq = 0.0f;

  // Boussinesq approximation of body force
  Fup = 0.055555555555555552f * gBetta * rho * (1 - vz) * (T - Tref) *
        (-1.5f * (vx) * (vx)-1.5f * (vy) * (vy) + 3.0f * (vz) * (vz) +
         3.0f * vz + 1);
  Fdown = 0.055555555555555552f * gBetta * rho * (T - Tref) * (-vz - 1) *
          (-1.5f * (vx) * (vx)-1.5f * (vy) * (vy) +
           3.0f * (vz) * (vz)-3.0f * vz + 1);

  // Modified heat diffusion
  tau_T = 1.0f / (5.0f * nuT + 0.5f + 5.0f * ST / Pr_t);

  // Modified shear viscosity
  tau_V = 1.0f / (3.0f * ST + 3.0f * nu + 0.5f);

  // Relax velocity
  omega0 = -5.1221899671258571e-18f * m11diff * tau_V -
           3.5855329769881e-18f * m12diff +
           2.9945110577043473e-17f * m14diff * tau_V +
           1.4508162637853633e-17f * m17diff - 0.01491228070175438f * m1diff +
           0.066666666666666652f * m2diff + 2.3500129416017396e-18f * m4diff +
           4.1085781098264214e-18f * m6diff - 4.142266011081772e-18f * m8diff;
  omega1 = -0.077777777777777765f * m10diff +
           1.0796062853147025e-17f * m11diff * tau_V +
           7.5572439972029167e-18f * m12diff -
           4.8979347812399405e-17f * m13diff * tau_V +
           2.3404530464533751e-17f * m14diff * tau_V +
           2.0983336949238271e-17f * m17diff - 0.0054678362573099401f * m1diff -
           0.022222222222222216f * m2diff - 0.11999999999999997f * m4diff -
           1.0913714530256704e-19f * m6diff +
           0.055555555555555525f * m9diff * tau_V;
  omega2 = -0.077777777777777779f * m10diff -
           1.260846761138673e-17f * m11diff * tau_V -
           8.825927327970711e-18f * m12diff + 4.415382210912669e-17f * m17diff -
           0.0054678362573099401f * m1diff - 0.02222222222222222f * m2diff +
           0.12f * m4diff + 2.6994065104414823e-18f * m6diff +
           6.3738607054330304e-18f * m8diff +
           0.055555555555555559f * m9diff * tau_V;
  omega3 =
      0.038888888888888876f * m10diff +
      0.083333333333333343f * m11diff * tau_V - 0.11666666666666667f * m12diff +
      2.9181704153691071e-17f * m14diff * tau_V +
      8.4984057833161437e-18f * m16diff + 1.5263934996972519e-17f * m17diff -
      0.0054678362573099427f * m1diff - 0.022222222222222209f * m2diff +
      5.6855286582580156e-18f * m4diff - 0.11999999999999997f * m6diff -
      4.0366650492166193e-18f * m8diff - 0.02777777777777778f * m9diff * tau_V;
  omega4 =
      0.038888888888888883f * m10diff +
      0.083333333333333315f * m11diff * tau_V - 0.11666666666666667f * m12diff +
      1.0772152509858619e-17f * m14diff * tau_V +
      1.2548427289427742e-17f * m16diff + 1.4627792700497276e-17f * m17diff -
      0.0054678362573099401f * m1diff - 0.022222222222222216f * m2diff -
      1.9473552611339476e-19f * m4diff + 0.11999999999999997f * m6diff -
      0.027777777777777776f * m9diff * tau_V;
  omega5 = 0.038888888888888876f * m10diff -
           0.083333333333333356f * m11diff * tau_V +
           0.1166666666666667f * m12diff + 7.9060005800095328e-17f * m17diff -
           0.0054678362573099418f * m1diff - 0.022222222222222216f * m2diff -
           3.377478152825545e-19f * m4diff + 1.0909316969586368e-17f * m6diff -
           0.11999999999999998f * m8diff -
           0.027777777777777759f * m9diff * tau_V;
  omega6 = 0.038888888888888896f * m10diff -
           0.083333333333333329f * m11diff * tau_V +
           0.11666666666666671f * m12diff -
           7.3589899908434058e-18f * m14diff * tau_V +
           2.5668641271963704e-17f * m17diff - 0.0054678362573099392f * m1diff -
           0.022222222222222223f * m2diff + 2.6525719270284167e-18f * m4diff +
           4.4376364207219304e-18f * m6diff + 0.12f * m8diff -
           0.027777777777777783f * m9diff * tau_V;
  omega7 = 0.083333333333333315f * m11diff * tau_V +
           0.058333333333333313f * m12diff + 0.25f * m13diff * tau_V -
           0.15081088650251664f * m14diff * tau_V + 0.2475f * m16diff -
           0.098197222362508538f * m17diff + 0.0025859521550944667f * m1diff +
           0.0091268899591569405f * m2diff + 0.029999999999999988f * m4diff +
           0.048097306380301991f * m6diff;
  omega8 = -0.014092681177017631f * m10diff +
           0.083333333333333315f * m11diff * tau_V +
           0.058333333333333313f * m12diff - 0.25f * m13diff * tau_V -
           0.087324231652879378f * m14diff * tau_V - 0.2475f * m16diff -
           0.16104901066364949f * m17diff + 0.0044920421251743699f * m1diff +
           0.0088079257356360185f * m2diff - 0.029999999999999992f * m4diff +
           0.040478907798345516f * m6diff - 0.02415888201774451f * m8diff -
           0.020132401681453759f * m9diff * tau_V;
  omega9 = 0.083333333333333287f * m11diff * tau_V +
           0.0583333333333333f * m12diff - 0.25f * m13diff * tau_V +
           0.15081088650251664f * m14diff * tau_V + 0.2475f * m16diff +
           0.098197222362508593f * m17diff + 0.0051719043101889333f * m1diff +
           0.0091268899591569405f * m2diff + 0.030000000000000009f * m4diff -
           0.04809730638030197f * m6diff - 0.020861462763787291f * m8diff;
  omega10 = 0.026156342665804279f * m10diff +
            0.083333333333333329f * m11diff * tau_V +
            0.058333333333333327f * m12diff + 0.25f * m13diff * tau_V -
            0.2475f * m16diff + 0.24750000000000005f * m17diff +
            0.0055582228164834093f * m1diff + 0.0065390856664510697f * m2diff -
            0.030000000000000009f * m4diff - 0.029999999999999995f * m6diff +
            0.03736620380829183f * m9diff * tau_V;
  omega11 = 0.019444444444444434f * m10diff -
            0.083333333333333343f * m11diff * tau_V -
            0.058333333333333334f * m12diff +
            4.1719532592455084e-17f * m14diff * tau_V +
            0.25f * m15diff * tau_V - 0.2475f * m16diff +
            4.4153822109126696e-17f * m17diff + 0.2475f * m18diff +
            0.0039766081871345036f * m1diff + 0.0055555555555555584f * m2diff +
            0.029999999999999999f * m4diff + 4.0256924739556932e-18f * m6diff +
            0.029999999999999995f * m8diff +
            0.027777777777777766f * m9diff * tau_V;
  omega12 = 0.019444444444444455f * m10diff -
            0.083333333333333356f * m11diff * tau_V -
            0.058333333333333348f * m12diff +
            5.3747365740451252e-17f * m14diff * tau_V -
            0.25f * m15diff * tau_V + 0.2475f * m16diff +
            4.4153822109126702e-17f * m17diff + 0.24749999999999994f * m18diff +
            0.0039766081871345036f * m1diff + 0.0055555555555555584f * m2diff -
            0.029999999999999999f * m4diff + 4.0256924739556924e-18f * m6diff +
            0.029999999999999999f * m8diff +
            0.027777777777777794f * m9diff * tau_V;
  omega13 = 0.019444444444444441f * m10diff -
            0.083333333333333343f * m11diff * tau_V -
            0.058333333333333334f * m12diff -
            5.1365806816332341e-17f * m14diff * tau_V -
            0.25f * m15diff * tau_V - 0.2475f * m16diff +
            5.0665441734426357e-18f * m17diff - 0.2475f * m18diff +
            0.0039766081871345019f * m1diff + 0.0055555555555555549f * m2diff +
            0.029999999999999995f * m4diff - 7.1215939703631473e-19f * m6diff -
            0.029999999999999995f * m8diff +
            0.027777777777777776f * m9diff * tau_V;
  omega14 = 0.019444444444444441f * m10diff -
            0.083333333333333356f * m11diff * tau_V -
            0.058333333333333348f * m12diff -
            1.9763270537949507e-17f * m14diff * tau_V +
            0.25f * m15diff * tau_V + 0.2475f * m16diff +
            8.3519746279953651e-18f * m17diff - 0.2475f * m18diff +
            0.0039766081871345019f * m1diff + 0.0055555555555555558f * m2diff -
            0.029999999999999999f * m4diff - 3.1392540254507416e-19f * m6diff -
            0.029999999999999999f * m8diff +
            0.027777777777777776f * m9diff * tau_V;
  omega15 = -0.040345849342932491f * m10diff -
            1.9196953237963454e-17f * m11diff * tau_V -
            1.3437867266574417e-17f * m12diff + 0.25f * m14diff * tau_V +
            0.24750000000000005f * m17diff - 0.45648446685146477f * m18diff +
            0.0042867464926865774f * m1diff + 0.0050432311678665614f * m2diff +
            0.029999999999999999f * m6diff + 0.069164313159312846f * m8diff -
            0.057636927632760702f * m9diff * tau_V;
  omega16 = -0.045623442528044476f * m10diff -
            2.170808415312752e-17f * m11diff * tau_V -
            1.5195658907189263e-17f * m12diff -
            0.28270220649126382f * m14diff * tau_V -
            0.21512481557364876f * m17diff + 0.0048474907686047262f * m1diff +
            0.0057029303160055595f * m2diff - 8.6832336612510068e-18f * m4diff -
            0.026075735221048342f * m6diff + 0.039105807881180982f * m8diff -
            0.065176346468634974f * m9diff * tau_V;
  omega17 = -0.034027777777777789f * m10diff -
            6.0055974811814068e-18f * m11diff * tau_V -
            4.2039182368269847e-18f * m12diff -
            0.27109356174493215f * m14diff * tau_V +
            0.26838262612748287f * m17diff + 0.495f * m18diff +
            0.0033047788742690079f * m1diff + 0.0056423611111111136f * m2diff -
            3.9847352127096138e-18f * m4diff + 0.032531227409391857f * m6diff +
            0.014999999999999999f * m8diff -
            0.048611111111111126f * m9diff * tau_V;
  omega18 = -0.043749999999999997f * m10diff -
            6.9388939039072284e-18f * m11diff * tau_V -
            4.8572257327350596e-18f * m12diff +
            0.27109356174493215f * m14diff * tau_V -
            0.26838262612748276f * m17diff + 0.0046484374999999998f * m1diff +
            0.0054687499999999997f * m2diff + 1.6653345369377347e-17f * m4diff -
            0.032531227409391843f * m6diff - 0.074999999999999997f * m8diff -
            0.0625f * m9diff * tau_V;

  // Difference to temperature equilibrium
  n1diff = T1 - T2 - n1eq;
  n2diff = T3 - T4 - n2eq;
  n3diff = T5 - T6 - n3eq;
  n4diff = 6.0f * T0 - 1.0f * T1 - 1.0f * T2 - 1.0f * T3 - 1.0f * T4 -
           1.0f * T5 - 1.0f * T6 - n4eq;
  n5diff = 2.0f * T1 + 2.0f * T2 - T3 - T4 - T5 - T6 - n5eq;
  n6diff = T3 + T4 - T5 - T6 - n6eq;

  // Relax temperature
  omegaT0 = 0.14285714285714285f * n4diff;
  omegaT1 = 0.5f * n1diff * tau_T - 0.023809523809523808f * n4diff +
            0.16666666666666666f * n5diff;
  omegaT2 = -0.5f * n1diff * tau_T - 0.023809523809523808f * n4diff +
            0.16666666666666666f * n5diff;
  omegaT3 = 0.5f * n2diff * tau_T - 0.023809523809523808f * n4diff -
            0.083333333333333329f * n5diff + 0.25f * n6diff;
  omegaT4 = -0.5f * n2diff * tau_T - 0.023809523809523808f * n4diff -
            0.083333333333333329f * n5diff + 0.25f * n6diff;
  omegaT5 = 0.5f * n3diff * tau_T - 0.023809523809523808f * n4diff -
            0.083333333333333329f * n5diff - 0.25f * n6diff;
  omegaT6 = -0.5f * n3diff * tau_T - 0.023809523809523808f * n4diff -
            0.083333333333333329f * n5diff - 0.25f * n6diff;

  // Write relaxed velocity
  dftmp3D(0, x, y, z, nx, ny, nz) = f0 - omega0;
  dftmp3D(1, x, y, z, nx, ny, nz) = f1 - omega1;
  dftmp3D(2, x, y, z, nx, ny, nz) = f2 - omega2;
  dftmp3D(3, x, y, z, nx, ny, nz) = f3 - omega3;
  dftmp3D(4, x, y, z, nx, ny, nz) = f4 - omega4;
  dftmp3D(5, x, y, z, nx, ny, nz) = Fup + f5 - omega5;
  dftmp3D(6, x, y, z, nx, ny, nz) = Fdown + f6 - omega6;
  dftmp3D(7, x, y, z, nx, ny, nz) = f7 - omega7;
  dftmp3D(8, x, y, z, nx, ny, nz) = f8 - omega8;
  dftmp3D(9, x, y, z, nx, ny, nz) = f9 - omega9;
  dftmp3D(10, x, y, z, nx, ny, nz) = f10 - omega10;
  dftmp3D(11, x, y, z, nx, ny, nz) = f11 - omega11;
  dftmp3D(12, x, y, z, nx, ny, nz) = f12 - omega12;
  dftmp3D(13, x, y, z, nx, ny, nz) = f13 - omega13;
  dftmp3D(14, x, y, z, nx, ny, nz) = f14 - omega14;
  dftmp3D(15, x, y, z, nx, ny, nz) = f15 - omega15;
  dftmp3D(16, x, y, z, nx, ny, nz) = f16 - omega16;
  dftmp3D(17, x, y, z, nx, ny, nz) = f17 - omega17;
  dftmp3D(18, x, y, z, nx, ny, nz) = f18 - omega18;

  // Write relaxed temperature
  Tdftmp3D(0, x, y, z, nx, ny, nz) = T0 - omegaT0;
  Tdftmp3D(1, x, y, z, nx, ny, nz) = T1 - omegaT1;
  Tdftmp3D(2, x, y, z, nx, ny, nz) = T2 - omegaT2;
  Tdftmp3D(3, x, y, z, nx, ny, nz) = T3 - omegaT3;
  Tdftmp3D(4, x, y, z, nx, ny, nz) = T4 - omegaT4;
  Tdftmp3D(5, x, y, z, nx, ny, nz) = T5 - omegaT5;
  Tdftmp3D(6, x, y, z, nx, ny, nz) = T6 - omegaT6;

  // Store macroscopic values
  phy->rho = rho;
  phy->T = T;
  phy->vx = vx;
  phy->vy = vy;
  phy->vz = vz;
}
