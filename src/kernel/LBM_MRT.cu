#include "LBM-MRT.h"
__device__ void computeMRT(int x,
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
  real m0eq, m1eq, m2eq, m3eq, m4eq, m5eq, m6eq, m7eq, m8eq, m9eq, m10eq, m11eq,
      m12eq, m13eq, m14eq, m15eq, m16eq, m17eq, m18eq;
  real m0diff, m1diff, m2diff, m3diff, m4diff, m5diff, m6diff, m7diff, m8diff,
      m9diff, m10diff, m11diff, m12diff, m13diff, m14diff, m15diff, m16diff,
      m17diff, m18diff;
  real omega0, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8,
      omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16,
      omega17, omega18;
  real n0eq, n1eq, n2eq, n3eq, n4eq, n5eq, n6eq;
  real n0diff, n1diff, n2diff, n3diff, n4diff, n5diff, n6diff;
  real omegaT0, omegaT1, omegaT2, omegaT3, omegaT4, omegaT5, omegaT6;
  real S_bar;
  real ST;
  real T;
  real Fup;
  real Fdown;
  real tau_V;
  real tau_T;
  real Sxx, Syy, Szz, Sxy, Syz, Sxz;
  real rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx3, pixx3, pww, piww, pxy,
      pyz, pxz, mx, my, mz;
  real omega_e, omega_xx, omega_ej;
  real vx, vy, vz;

  // Macroscopic density
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
        f3 + f4 + f5 + f6 + f7 + f8 + f9;

  // Part of kinetic energy independent of density
  en = -30.0 * f0 - 11.0 * f1 + 8.0 * f10 + 8.0 * f11 + 8.0 * f12 + 8.0 * f13 +
       8.0 * f14 + 8.0 * f15 + 8.0 * f16 + 8.0 * f17 + 8.0 * f18 - 11.0 * f2 -
       11.0 * f3 - 11.0 * f4 - 11.0 * f5 - 11.0 * f6 + 8.0 * f7 + 8.0 * f8 +
       8.0 * f9;

  // Kinetic energy square
  epsilon = 12.0 * f0 - 4.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 +
            1.0 * f13 + 1.0 * f14 + 1.0 * f15 + 1.0 * f16 + 1.0 * f17 +
            1.0 * f18 - 4.0 * f2 - 4.0 * f3 - 4.0 * f4 - 4.0 * f5 - 4.0 * f6 +
            1.0 * f7 + 1.0 * f8 + 1.0 * f9;

  // Momentum
  jx = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
  jy = f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9;
  jz = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6;

  // Energy flux independent of mass flux
  qx = -4.0 * f1 - 1.0 * f10 + 1.0 * f11 - 1.0 * f12 + 1.0 * f13 - 1.0 * f14 +
       4.0 * f2 + 1.0 * f7 - 1.0 * f8 + 1.0 * f9;
  qy = 1.0 * f10 + 1.0 * f15 - 1.0 * f16 + 1.0 * f17 - 1.0 * f18 - 4.0 * f3 +
       4.0 * f4 + 1.0 * f7 - 1.0 * f8 - 1.0 * f9;
  qz = 1.0 * f11 - 1.0 * f12 - 1.0 * f13 + 1.0 * f14 + 1.0 * f15 - 1.0 * f16 -
       1.0 * f17 + 1.0 * f18 - 4.0 * f5 + 4.0 * f6;

  // Symmetric traceless viscous stress tensor
  pxx3 = 2.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 + 1.0 * f13 + 1.0 * f14 -
         2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0 * f2 - f3 - f4 - f5 - f6 +
         1.0 * f7 + 1.0 * f8 + 1.0 * f9;
  pww = f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9;
  pxy = -f10 + f7 + f8 - f9;
  pyz = f15 + f16 - f17 - f18;
  pxz = f11 + f12 - f13 - f14;

  // Fourth order moments
  pixx3 = -4.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 + 1.0 * f13 +
          1.0 * f14 - 2.0 * f15 - 2.0 * f16 - 2.0 * f17 - 2.0 * f18 - 4.0 * f2 +
          2.0 * f3 + 2.0 * f4 + 2.0 * f5 + 2.0 * f6 + 1.0 * f7 + 1.0 * f8 +
          1.0 * f9;
  piww = 1.0 * f10 - 1.0 * f11 - 1.0 * f12 - 1.0 * f13 - 1.0 * f14 - 2.0 * f3 -
         2.0 * f4 + 2.0 * f5 + 2.0 * f6 + 1.0 * f7 + 1.0 * f8 + 1.0 * f9;

  // Antisymmetric third-order moment
  mx = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9;
  my = -f10 + f15 - f16 + f17 - f18 - f7 + f8 + f9;
  mz = f11 - f12 - f13 + f14 - f15 + f16 + f17 - f18;

  // Model stability constants
  omega_e = 0;
  omega_xx = 0;
  omega_ej = -7.5396825396825395;

  // Macroscopic velocity
  vx = jx / rho;
  vy = jy / rho;
  vz = jz / rho;

  // Velocity moment equilibirum distribution functions
  m0eq = rho;
  m1eq = 19.0 * pow(jx, 2) + 19.0 * pow(jy, 2) + 19.0 * pow(jz, 2) - 11.0 * rho;
  m2eq =
      omega_e * rho + 1.0 * omega_ej * (pow(jx, 2) + pow(jy, 2) + pow(jz, 2));
  m3eq = jx;
  m4eq = -0.66666666666666663 * jx;
  m5eq = jy;
  m6eq = -0.66666666666666663 * jy;
  m7eq = jz;
  m8eq = -0.66666666666666663 * jz;
  m9eq = 2.0 * pow(jx, 2) - 1.0 * pow(jy, 2) - 1.0 * pow(jz, 2);
  m10eq = omega_xx * (2.0 * pow(jx, 2) - 1.0 * pow(jy, 2) - 1.0 * pow(jz, 2));
  m11eq = 1.0 * pow(jy, 2) - 1.0 * pow(jz, 2);
  m12eq = omega_xx * (1.0 * pow(jy, 2) - 1.0 * pow(jz, 2));
  m13eq = 1.0 * jx * jy;
  m14eq = 1.0 * jy * jz;
  m15eq = 1.0 * jx * jz;
  m16eq = 0;
  m17eq = 0;
  m18eq = 0;

  // Difference to velocity equilibrium
  m0diff = -m0eq + rho;
  m1diff = en - m1eq;
  m2diff = epsilon - m2eq;
  m3diff = jx - m3eq;
  m4diff = -m4eq + qx;
  m5diff = jy - m5eq;
  m6diff = -m6eq + qy;
  m7diff = jz - m7eq;
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
  Sxx = 0.93947368421052613 * f0 + 0.34447368421052627 * f1 -
        0.25052631578947365 * f10 - 0.25052631578947365 * f11 -
        0.25052631578947365 * f12 - 0.25052631578947365 * f13 -
        0.25052631578947365 * f14 - 0.25052631578947365 * f15 -
        0.25052631578947365 * f16 - 0.25052631578947365 * f17 -
        0.25052631578947365 * f18 + 0.34447368421052627 * f2 +
        0.34447368421052627 * f3 + 0.34447368421052627 * f4 +
        0.34447368421052627 * f5 + 0.34447368421052627 * f6 -
        0.25052631578947365 * f7 - 0.25052631578947365 * f8 -
        0.25052631578947365 * f9 -
        0.5 *
            (2.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 + 1.0 * f13 +
             1.0 * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0 * f2 - f3 -
             f4 - f5 - f6 + 1.0 * f7 + 1.0 * f8 + 1.0 * f9) /
            (3.0 * nu + 0.5);
  Syy = 0.93947368421052613 * f0 + 0.34447368421052627 * f1 -
        0.25052631578947365 * f10 - 0.25052631578947365 * f11 -
        0.25052631578947365 * f12 - 0.25052631578947365 * f13 -
        0.25052631578947365 * f14 - 0.25052631578947365 * f15 -
        0.25052631578947365 * f16 - 0.25052631578947365 * f17 -
        0.25052631578947365 * f18 + 0.34447368421052627 * f2 +
        0.34447368421052627 * f3 + 0.34447368421052627 * f4 +
        0.34447368421052627 * f5 + 0.34447368421052627 * f6 -
        0.25052631578947365 * f7 - 0.25052631578947365 * f8 -
        0.25052631578947365 * f9 +
        0.25 *
            (2.0 * f1 - 2.0 * f10 + 4.0 * f11 + 4.0 * f12 + 4.0 * f13 +
             4.0 * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0 * f2 -
             4.0 * f3 - 4.0 * f4 + 2.0 * f5 + 2.0 * f6 - 2.0 * f7 - 2.0 * f8 -
             2.0 * f9) /
            (3.0 * nu + 0.5);
  Szz = 0.93947368421052613 * f0 + 0.34447368421052627 * f1 -
        0.25052631578947365 * f10 - 0.25052631578947365 * f11 -
        0.25052631578947365 * f12 - 0.25052631578947365 * f13 -
        0.25052631578947365 * f14 - 0.25052631578947365 * f15 -
        0.25052631578947365 * f16 - 0.25052631578947365 * f17 -
        0.25052631578947365 * f18 + 0.34447368421052627 * f2 +
        0.34447368421052627 * f3 + 0.34447368421052627 * f4 +
        0.34447368421052627 * f5 + 0.34447368421052627 * f6 -
        0.25052631578947365 * f7 - 0.25052631578947365 * f8 -
        0.25052631578947365 * f9 +
        0.25 *
            (2.0 * f1 + 4.0 * f10 - 2.0 * f11 - 2.0 * f12 - 2.0 * f13 -
             2.0 * f14 - 2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0 * f2 +
             2.0 * f3 + 2.0 * f4 - 4.0 * f5 - 4.0 * f6 + 4.0 * f7 + 4.0 * f8 +
             4.0 * f9) /
            (3.0 * nu + 0.5);
  Sxy = -1.5 * (-f10 + f7 + f8 - f9) / (3.0 * nu + 0.5);
  Syz = -1.5 * (f15 + f16 - f17 - f18) / (3.0 * nu + 0.5);
  Sxz = -1.5 * (f11 + f12 - f13 - f14) / (3.0 * nu + 0.5);

  // Magnitude of strain rate tensor
  S_bar = sqrt(2.0 * pow(Sxx, 2) + 4.0 * pow(Sxy, 2) + 4.0 * pow(Sxz, 2) +
               2.0 * pow(Syy, 2) + 4.0 * pow(Syz, 2) + 2.0 * pow(Szz, 2));

  // Filtered strain rate
  ST = 4.0 * pow(C, 2) * S_bar;

  // Macroscopic temperature
  T = T0 + T1 + T2 + T3 + T4 + T5 + T6;

  // Temperature moment equilibirum distribution functions
  n0eq = T;
  n1eq = T * vx;
  n2eq = T * vy;
  n3eq = T * vz;
  n4eq = 0.75 * T;
  n5eq = 0.0;
  n6eq = 0.0;

  // Boussinesq approximation of body force
  Fup =
      0.055555555555555552 * gBetta * rho * (1 - vz) * (T - Tref) *
      (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) + 3.0 * pow(vz, 2) + 3.0 * vz + 1);
  Fdown =
      0.055555555555555552 * gBetta * rho * (T - Tref) * (-vz - 1) *
      (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) + 3.0 * pow(vz, 2) - 3.0 * vz + 1);

  // Modified heat diffusion
  tau_T = 1.0 / (5.0 * nuT + 0.5 + 5.0 * ST / Pr_t);

  // Modified shear viscosity
  tau_V = 1.0 / (3.0 * ST + 3.0 * nu + 0.5);

  // Relax velocity
  omega0 = 7.0821054079792582e-18 * m10diff -
           2.401486436084884e-17 * m11diff * tau_V -
           1.6810405052594186e-17 * m12diff - 0.014912280701754384 * m1diff +
           0.066666666666666652 * m2diff - 3.6022296541273257e-18 * m4diff +
           6.0703760639822214e-18 * m8diff +
           1.0117293439970369e-17 * m9diff * tau_V;
  omega1 = -0.077777777777777779 * m10diff -
           2.4014864360848846e-17 * m11diff * tau_V -
           1.6810405052594192e-17 * m12diff -
           1.7516143225401517e-17 * m14diff * tau_V -
           3.8298197536040582e-17 * m16diff - 1.73409817931475e-17 * m17diff -
           1.73409817931475e-17 * m18diff - 0.0054678362573099409 * m1diff -
           0.02222222222222222 * m2diff - 0.11999999999999997 * m4diff -
           2.1019371870481824e-18 * m6diff + 2.1019371870481824e-18 * m8diff +
           0.05555555555555558 * m9diff * tau_V;
  omega2 = -0.077777777777777793 * m10diff -
           4.9745672903741283e-18 * m11diff * tau_V -
           3.48219710326189e-18 * m12diff -
           1.8398708348606641e-17 * m14diff * tau_V +
           3.5284750217750149e-17 * m17diff - 1.8214721265120573e-17 * m18diff -
           0.0054678362573099444 * m1diff - 0.02222222222222223 * m2diff +
           0.12 * m4diff + 3.0870522256950994e-19 * m6diff +
           3.8290411073743348e-18 * m8diff +
           0.05555555555555558 * m9diff * tau_V;
  omega3 = 0.038888888888888903 * m10diff +
           0.083333333333333315 * m11diff * tau_V -
           0.1166666666666667 * m12diff - 0.0054678362573099401 * m1diff -
           0.022222222222222223 * m2diff - 4.8284862705207176e-18 * m4diff -
           0.12000000000000002 * m6diff + 3.4803421458065534e-18 * m8diff -
           0.027777777777777766 * m9diff * tau_V;
  omega4 = 0.038888888888888896 * m10diff +
           0.083333333333333329 * m11diff * tau_V -
           0.11666666666666667 * m12diff - 0.0054678362573099418 * m1diff -
           0.02222222222222222 * m2diff - 5.4128741892706893e-18 * m4diff +
           0.11999999999999998 * m6diff + 5.1389426996674887e-18 * m8diff -
           0.027777777777777742 * m9diff * tau_V;
  omega5 =
      0.038888888888888896 * m10diff - 0.08333333333333337 * m11diff * tau_V +
      0.11666666666666663 * m12diff - 1.4164799996519842e-17 * m15diff * tau_V -
      0.0054678362573099409 * m1diff - 0.022222222222222223 * m2diff -
      9.9763896525612572e-18 * m4diff - 0.12000000000000002 * m8diff -
      0.027777777777777773 * m9diff * tau_V;
  omega6 = 0.038888888888888883 * m10diff -
           0.083333333333333356 * m11diff * tau_V +
           0.11666666666666665 * m12diff - 0.0054678362573099392 * m1diff -
           0.02222222222222222 * m2diff - 3.6022296541273273e-18 * m4diff +
           0.11999999999999998 * m8diff - 0.027777777777777755 * m9diff * tau_V;
  omega7 =
      0.019444444444444431 * m10diff + 0.083333333333333315 * m11diff * tau_V +
      0.058333333333333313 * m12diff + 0.25 * m13diff * tau_V +
      0.2475 * m16diff - 0.2475 * m17diff + 0.0039766081871345027 * m1diff +
      0.0055555555555555558 * m2diff + 0.029999999999999975 * m4diff +
      0.030000000000000006 * m6diff + 8.5632953907980939e-18 * m8diff +
      0.027777777777777762 * m9diff * tau_V;
  omega8 =
      0.019444444444444455 * m10diff + 0.083333333333333315 * m11diff * tau_V +
      0.058333333333333313 * m12diff + 0.25 * m13diff * tau_V -
      0.2475 * m16diff + 0.2475 * m17diff + 0.0039766081871345045 * m1diff +
      0.0055555555555555575 * m2diff - 0.029999999999999995 * m4diff -
      0.030000000000000009 * m6diff + 5.3716100862084035e-18 * m8diff +
      0.027777777777777797 * m9diff * tau_V;
  omega9 =
      0.019444444444444448 * m10diff + 0.083333333333333315 * m11diff * tau_V +
      0.058333333333333313 * m12diff - 0.25 * m13diff * tau_V +
      0.2475 * m16diff + 0.2475 * m17diff + 0.0039766081871345019 * m1diff +
      0.005555555555555554 * m2diff + 0.029999999999999985 * m4diff -
      0.029999999999999988 * m6diff + 0.027777777777777783 * m9diff * tau_V;
  omega10 =
      0.019444444444444448 * m10diff + 0.083333333333333329 * m11diff * tau_V +
      0.058333333333333327 * m12diff - 0.25 * m13diff * tau_V -
      0.2475 * m16diff - 0.2475 * m17diff + 0.0039766081871345027 * m1diff +
      0.0055555555555555558 * m2diff - 0.029999999999999992 * m4diff +
      0.029999999999999995 * m6diff + 1.0032289829428092e-17 * m8diff +
      0.027777777777777787 * m9diff * tau_V;
  omega11 =
      0.019444444444444455 * m10diff - 0.08333333333333337 * m11diff * tau_V -
      0.058333333333333355 * m12diff + 0.25 * m15diff * tau_V -
      0.2475 * m16diff + 0.2475 * m18diff + 0.0039766081871345027 * m1diff +
      0.0055555555555555549 * m2diff + 0.029999999999999985 * m4diff +
      0.029999999999999999 * m8diff + 0.027777777777777797 * m9diff * tau_V;
  omega12 =
      0.019444444444444441 * m10diff - 0.083333333333333343 * m11diff * tau_V -
      0.058333333333333334 * m12diff + 0.25 * m15diff * tau_V +
      0.2475 * m16diff - 0.2475 * m18diff + 0.0039766081871345027 * m1diff +
      0.0055555555555555558 * m2diff - 0.030000000000000009 * m4diff -
      0.029999999999999999 * m8diff + 0.027777777777777776 * m9diff * tau_V;
  omega13 =
      0.019444444444444434 * m10diff - 0.083333333333333329 * m11diff * tau_V -
      0.058333333333333327 * m12diff - 0.25 * m15diff * tau_V -
      0.2475 * m16diff - 0.2475 * m18diff + 0.0039766081871345027 * m1diff +
      0.0055555555555555558 * m2diff + 0.029999999999999988 * m4diff -
      0.029999999999999999 * m8diff + 0.027777777777777766 * m9diff * tau_V;
  omega14 =
      0.019444444444444448 * m10diff - 0.083333333333333356 * m11diff * tau_V -
      0.058333333333333348 * m12diff - 0.25 * m15diff * tau_V +
      0.2475 * m16diff + 0.2475 * m18diff + 0.0039766081871345036 * m1diff +
      0.0055555555555555558 * m2diff - 0.030000000000000009 * m4diff +
      0.030000000000000009 * m8diff + 0.027777777777777787 * m9diff * tau_V;
  omega15 = -0.038888888888888869 * m10diff -
            1.7731162223744187e-17 * m11diff * tau_V -
            1.241181355662093e-17 * m12diff + 0.25 * m14diff * tau_V +
            0.2475 * m17diff - 0.2475 * m18diff +
            0.0039766081871345036 * m1diff + 0.0055555555555555558 * m2diff -
            5.4873402952587282e-18 * m4diff + 0.029999999999999995 * m6diff +
            0.030000000000000006 * m8diff -
            0.055555555555555532 * m9diff * tau_V;
  omega16 = -0.038888888888888869 * m10diff -
            3.0298566497953508e-17 * m11diff * tau_V -
            2.1208996548567455e-17 * m12diff + 0.25 * m14diff * tau_V -
            0.2475 * m17diff + 0.2475 * m18diff +
            0.0039766081871345027 * m1diff + 0.0055555555555555558 * m2diff -
            1.7171190129959261e-18 * m4diff - 0.029999999999999995 * m6diff -
            0.029999999999999975 * m8diff -
            0.055555555555555532 * m9diff * tau_V;
  omega17 = -0.038888888888888869 * m10diff -
            3.0298566497953527e-17 * m11diff * tau_V -
            2.1208996548567468e-17 * m12diff - 0.25 * m14diff * tau_V +
            0.2475 * m17diff + 0.2475 * m18diff +
            0.0039766081871345036 * m1diff + 0.0055555555555555558 * m2diff -
            1.7171190129959275e-18 * m4diff + 0.029999999999999995 * m6diff -
            0.029999999999999992 * m8diff -
            0.055555555555555532 * m9diff * tau_V;
  omega18 = -0.038888888888888883 * m10diff -
            1.7731162223744184e-17 * m11diff * tau_V -
            1.2411813556620928e-17 * m12diff - 0.25 * m14diff * tau_V -
            0.2475 * m17diff - 0.2475 * m18diff +
            0.0039766081871345027 * m1diff + 0.0055555555555555558 * m2diff -
            5.4873402952587267e-18 * m4diff - 0.029999999999999995 * m6diff +
            0.029999999999999999 * m8diff -
            0.055555555555555546 * m9diff * tau_V;

  // Difference to temperature equilibrium
  n0diff = T0 + T1 + T2 + T3 + T4 + T5 + T6 - n0eq;
  n1diff = T1 - T2 - n1eq;
  n2diff = T3 - T4 - n2eq;
  n3diff = T5 - T6 - n3eq;
  n4diff = 6.0 * T0 - 1.0 * T1 - 1.0 * T2 - 1.0 * T3 - 1.0 * T4 - 1.0 * T5 -
           1.0 * T6 - n4eq;
  n5diff = 2.0 * T1 + 2.0 * T2 - T3 - T4 - T5 - T6 - n5eq;
  n6diff = T3 + T4 - T5 - T6 - n6eq;

  // Relax temperature
  omegaT0 = 0.14285714285714285 * n4diff;
  omegaT1 = 0.5 * n1diff * tau_T - 0.023809523809523808 * n4diff +
            0.16666666666666666 * n5diff;
  omegaT2 = -0.5 * n1diff * tau_T - 0.023809523809523808 * n4diff +
            0.16666666666666666 * n5diff;
  omegaT3 = 0.5 * n2diff * tau_T - 0.023809523809523808 * n4diff -
            0.083333333333333329 * n5diff + 0.25 * n6diff;
  omegaT4 = -0.5 * n2diff * tau_T - 0.023809523809523808 * n4diff -
            0.083333333333333329 * n5diff + 0.25 * n6diff;
  omegaT5 = 0.5 * n3diff * tau_T - 0.023809523809523808 * n4diff -
            0.083333333333333329 * n5diff - 0.25 * n6diff;
  omegaT6 = -0.5 * n3diff * tau_T - 0.023809523809523808 * n4diff -
            0.083333333333333329 * n5diff - 0.25 * n6diff;

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
  phy->rho = rho;
  phy->T = T;
  phy->vx = vx;
  phy->vy = vy;
  phy->vz = vz;
}