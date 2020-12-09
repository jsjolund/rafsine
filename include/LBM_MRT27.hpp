#pragma once
#include "CudaUtils.hpp"
#include "PhysicalQuantity.hpp"
__device__ __forceinline__ void computeMRT27(int x,
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
                                             real_t f19,
                                             real_t f20,
                                             real_t f21,
                                             real_t f22,
                                             real_t f23,
                                             real_t f24,
                                             real_t f25,
                                             real_t f26,
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
  real_t m1eq, m2eq, m3eq, m4eq, m5eq, m6eq, m7eq, m8eq, m9eq, m10eq, m11eq,
      m12eq, m13eq, m14eq, m15eq, m16eq, m17eq, m18eq, m19eq, m20eq, m21eq,
      m22eq, m23eq, m24eq, m25eq, m26eq;
  real_t m1diff, m2diff, m3diff, m4diff, m5diff, m6diff, m7diff, m8diff, m9diff,
      m10diff, m11diff, m12diff, m13diff, m14diff, m15diff, m16diff, m17diff,
      m18diff, m19diff, m20diff, m21diff, m22diff, m23diff, m24diff, m25diff,
      m26diff;
  real_t omega0, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8,
      omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16,
      omega17, omega18, omega19, omega20, omega21, omega22, omega23, omega24,
      omega25, omega26;
  real_t S_bar;
  real_t ST;
  real_t T;
  real_t Fup;
  real_t Fdown;
  real_t tau_V;
  real_t tau_T;
  real_t Sxx, Syy, Szz, Sxy, Syz, Sxz;
  real_t rho, jx, jy, jz;
  real_t ux, uy, uz;
  real_t u_bar;
  real_t T0eq, T1eq, T2eq, T3eq, T4eq, T5eq, T6eq;

  // Macroscopic density
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19 +
        f2 + f20 + f21 + f22 + f23 + f24 + f25 + f26 + f3 + f4 + f5 + f6 + f7 +
        f8 + f9;

  // Momentum
  jx = f1 - f10 + f11 - f12 + f13 - f14 + f19 - f2 - f20 + f21 - f22 + f23 -
       f24 + f25 - f26 + f7 - f8 + f9;
  jy = -f10 + f15 - f16 + f17 - f18 + f19 + f20 - f21 - f22 + f23 + f24 - f25 -
       f26 + f3 - f4 + f7 + f8 - f9;
  jz = f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 + f19 + f20 + f21 + f22 -
       f23 - f24 - f25 - f26 + f5 - f6;

  // Macroscopic velocity
  ux = jx / rho;
  uy = jy / rho;
  uz = jz / rho;
  u_bar = powf((ux) * (ux) + (uy) * (uy) + (uz) * (uz), 0.5f);
  m1eq = -0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) +
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
              4.5f * (-ux - uy) * (-ux - uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
              4.5f * (-ux + uy) * (-ux + uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
              4.5f * (-ux - uz) * (-ux - uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
              4.5f * (-ux + uz) * (-ux + uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
              4.5f * (ux - uy) * (ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
              4.5f * (ux + uy) * (ux + uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
              4.5f * (ux - uz) * (ux - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
              4.5f * (ux + uz) * (ux + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m2eq = -0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) +
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
              4.5f * (-ux - uy) * (-ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
              4.5f * (-ux + uy) * (-ux + uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
              4.5f * (ux - uy) * (ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
              4.5f * (ux + uy) * (ux + uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
              4.5f * (-uy - uz) * (-uy - uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
              4.5f * (-uy + uz) * (-uy + uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
              4.5f * (uy - uz) * (uy - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
              4.5f * (uy + uz) * (uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m3eq = -0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) +
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
              4.5f * (-ux - uz) * (-ux - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
              4.5f * (-ux + uz) * (-ux + uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
              4.5f * (ux - uz) * (ux - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
              4.5f * (ux + uz) * (ux + uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
              4.5f * (-uy - uz) * (-uy - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
              4.5f * (-uy + uz) * (-uy + uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
              4.5f * (uy - uz) * (uy - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
              4.5f * (uy + uz) * (uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m4eq = -0.59259259259259256f * rho * (1 - 1.5f * (u_bar) * (u_bar)) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m5eq = 0.14814814814814814f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) +
         0.14814814814814814f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
              4.5f * (-ux - uy) * (-ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
              4.5f * (-ux + uy) * (-ux + uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
              4.5f * (-ux - uz) * (-ux - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
              4.5f * (-ux + uz) * (-ux + uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
              4.5f * (ux - uy) * (ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
              4.5f * (ux + uy) * (ux + uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
              4.5f * (ux - uz) * (ux - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
              4.5f * (ux + uz) * (ux + uz) + 1) -
         0.037037037037037035f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
              4.5f * (-uy - uz) * (-uy - uz) + 1) -
         0.037037037037037035f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
              4.5f * (-uy + uz) * (-uy + uz) + 1) -
         0.037037037037037035f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
              4.5f * (uy - uz) * (uy - uz) + 1) -
         0.037037037037037035f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
              4.5f * (uy + uz) * (uy + uz) + 1);
  m6eq = 0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) +
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) -
         0.07407407407407407f * rho *
             (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
              4.5f * (-ux - uy) * (-ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
              4.5f * (-ux + uy) * (-ux + uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
              4.5f * (-ux - uz) * (-ux - uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
              4.5f * (-ux + uz) * (-ux + uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
              4.5f * (ux - uy) * (ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
              4.5f * (ux + uy) * (ux + uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
              4.5f * (ux - uz) * (ux - uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
              4.5f * (ux + uz) * (ux + uz) + 1);
  m7eq = 0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
              4.5f * (-ux - uy) * (-ux - uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
              4.5f * (-ux + uy) * (-ux + uy) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
              4.5f * (ux - uy) * (ux - uy) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
              4.5f * (ux + uy) * (ux + uy) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m8eq = 0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
              4.5f * (-uy - uz) * (-uy - uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
              4.5f * (-uy + uz) * (-uy + uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
              4.5f * (uy - uz) * (uy - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
              4.5f * (uy + uz) * (uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m9eq = 0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
              4.5f * (-ux - uz) * (-ux - uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
              4.5f * (-ux + uz) * (-ux + uz) + 1) -
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
              4.5f * (ux - uz) * (ux - uz) + 1) +
         0.018518518518518517f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
              4.5f * (ux + uz) * (ux + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
              4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
              4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
              4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
         0.0046296296296296294f * rho *
             (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
              4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m10eq = 0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) -
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m11eq = 0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) -
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m12eq = 0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) -
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0092592592592592587f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m13eq = -0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m14eq = -0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m15eq = -0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m16eq = 1.1851851851851851f * rho * (1 - 1.5f * (u_bar) * (u_bar)) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m17eq = -2.3703703703703702f * rho * (1 - 1.5f * (u_bar) * (u_bar)) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) +
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m18eq = -0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux)-3.0f * ux + 1) -
          0.29629629629629628f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (ux) * (ux) + 3.0f * ux + 1) +
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) +
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) +
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) +
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1);
  m19eq = -0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy)-3.0f * uy + 1) -
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uy) * (uy) + 3.0f * uy + 1) +
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz)-3.0f * uz + 1) +
          0.14814814814814814f * rho *
              (-1.5f * (u_bar) * (u_bar) + 4.5f * (uz) * (uz) + 3.0f * uz + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1);
  m20eq = -0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m21eq = -0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m22eq = -0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) +
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.037037037037037035f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);
  m23eq = -0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1);
  m24eq = 0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy +
               4.5f * (-ux - uy) * (-ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy +
               4.5f * (-ux + uy) * (-ux + uy) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy +
               4.5f * (ux - uy) * (ux - uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy +
               4.5f * (ux + uy) * (ux + uy) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1);
  m25eq = -0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uz +
               4.5f * (-ux - uz) * (-ux - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uz +
               4.5f * (-ux + uz) * (-ux + uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uz +
               4.5f * (ux - uz) * (ux - uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uz +
               4.5f * (ux + uz) * (ux + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy - 3.0f * uz +
               4.5f * (-uy - uz) * (-uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * uy + 3.0f * uz +
               4.5f * (-uy + uz) * (-uy + uz) + 1) +
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy - 3.0f * uz +
               4.5f * (uy - uz) * (uy - uz) + 1) -
          0.018518518518518517f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * uy + 3.0f * uz +
               4.5f * (uy + uz) * (uy + uz) + 1);
  m26eq = -0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (-ux - uy - uz) * (-ux - uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (-ux - uy + uz) * (-ux - uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (-ux + uy - uz) * (-ux + uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar)-3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (-ux + uy + uz) * (-ux + uy + uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy - 3.0f * uz +
               4.5f * (ux - uy - uz) * (ux - uy - uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux - 3.0f * uy + 3.0f * uz +
               4.5f * (ux - uy + uz) * (ux - uy + uz) + 1) -
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy - 3.0f * uz +
               4.5f * (ux + uy - uz) * (ux + uy - uz) + 1) +
          0.0046296296296296294f * rho *
              (-1.5f * (u_bar) * (u_bar) + 3.0f * ux + 3.0f * uy + 3.0f * uz +
               4.5f * (ux + uy + uz) * (ux + uy + uz) + 1);

  // Difference to velocity equilibrium
  m1diff = f1 - f10 + f11 - f12 + f13 - f14 + f19 - f2 - f20 + f21 - f22 + f23 -
           f24 + f25 - f26 + f7 - f8 + f9 - m1eq;
  m2diff = -f10 + f15 - f16 + f17 - f18 + f19 + f20 - f21 - f22 + f23 + f24 -
           f25 - f26 + f3 - f4 + f7 + f8 - f9 - m2eq;
  m3diff = f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 + f19 + f20 + f21 +
           f22 - f23 - f24 - f25 - f26 + f5 - f6 - m3eq;
  m4diff = -2 * f0 - f1 + f19 - f2 + f20 + f21 + f22 + f23 + f24 + f25 + f26 -
           f3 - f4 - f5 - f6 - m4eq;
  m5diff = 2 * f1 + f10 + f11 + f12 + f13 + f14 - 2 * f15 - 2 * f16 - 2 * f17 -
           2 * f18 + 2 * f2 - f3 - f4 - f5 - f6 + f7 + f8 + f9 - m5eq;
  m6diff =
      f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9 - m6eq;
  m7diff =
      f10 + f19 - f20 - f21 + f22 + f23 - f24 - f25 + f26 + f7 - f8 - f9 - m7eq;
  m8diff = f15 - f16 - f17 + f18 + f19 + f20 - f21 - f22 - f23 - f24 + f25 +
           f26 - m8eq;
  m9diff = f11 - f12 - f13 + f14 + f19 - f20 + f21 - f22 - f23 + f24 - f25 +
           f26 - m9eq;
  m10diff = -4 * f1 + f10 - f11 + f12 - f13 + f14 + 2 * f19 + 4 * f2 - 2 * f20 +
            2 * f21 - 2 * f22 + 2 * f23 - 2 * f24 + 2 * f25 - 2 * f26 - f7 +
            f8 - f9 - m10eq;
  m11diff = f10 - f15 + f16 - f17 + f18 + 2 * f19 + 2 * f20 - 2 * f21 -
            2 * f22 + 2 * f23 + 2 * f24 - 2 * f25 - 2 * f26 - 4 * f3 + 4 * f4 -
            f7 - f8 + f9 - m11eq;
  m12diff = -f11 - f12 + f13 + f14 - f15 - f16 + f17 + f18 + 2 * f19 + 2 * f20 +
            2 * f21 + 2 * f22 - 2 * f23 - 2 * f24 - 2 * f25 - 2 * f26 - 4 * f5 +
            4 * f6 - m12eq;
  m13diff = 4.0f * f1 + 2.0f * f10 - 2.0f * f11 + 2.0f * f12 - 2.0f * f13 +
            2.0f * f14 + 1.0f * f19 - 4.0f * f2 - 1.0f * f20 + 1.0f * f21 -
            1.0f * f22 + 1.0f * f23 - 1.0f * f24 + 1.0f * f25 - 1.0f * f26 -
            2.0f * f7 + 2.0f * f8 - 2.0f * f9 - m13eq;
  m14diff = 2.0f * f10 - 2.0f * f15 + 2.0f * f16 - 2.0f * f17 + 2.0f * f18 +
            1.0f * f19 + 1.0f * f20 - 1.0f * f21 - 1.0f * f22 + 1.0f * f23 +
            1.0f * f24 - 1.0f * f25 - 1.0f * f26 + 4.0f * f3 - 4.0f * f4 -
            2.0f * f7 - 2.0f * f8 + 2.0f * f9 - m14eq;
  m15diff = -2.0f * f11 - 2.0f * f12 + 2.0f * f13 + 2.0f * f14 - 2.0f * f15 -
            2.0f * f16 + 2.0f * f17 + 2.0f * f18 + 1.0f * f19 + 1.0f * f20 +
            1.0f * f21 + 1.0f * f22 - 1.0f * f23 - 1.0f * f24 - 1.0f * f25 -
            1.0f * f26 + 4.0f * f5 - 4.0f * f6 - m15eq;
  m16diff = 4.0f * f0 - 1.0f * f10 - 1.0f * f11 - 1.0f * f12 - 1.0f * f13 -
            1.0f * f14 - 1.0f * f15 - 1.0f * f16 - 1.0f * f17 - 1.0f * f18 +
            1.0f * f19 + 1.0f * f20 + 1.0f * f21 + 1.0f * f22 + 1.0f * f23 +
            1.0f * f24 + 1.0f * f25 + 1.0f * f26 - 1.0f * f7 - 1.0f * f8 -
            1.0f * f9 - m16eq;
  m17diff = -8.0f * f0 + 4.0f * f1 - 2.0f * f10 - 2.0f * f11 - 2.0f * f12 -
            2.0f * f13 - 2.0f * f14 - 2.0f * f15 - 2.0f * f16 - 2.0f * f17 -
            2.0f * f18 + 1.0f * f19 + 4.0f * f2 + 1.0f * f20 + 1.0f * f21 +
            1.0f * f22 + 1.0f * f23 + 1.0f * f24 + 1.0f * f25 + 1.0f * f26 +
            4.0f * f3 + 4.0f * f4 + 4.0f * f5 + 4.0f * f6 - 2.0f * f7 -
            2.0f * f8 - 2.0f * f9 - m17eq;
  m18diff = -4 * f1 + f10 + f11 + f12 + f13 + f14 - 2 * f15 - 2 * f16 -
            2 * f17 - 2 * f18 - 4 * f2 + 2 * f3 + 2 * f4 + 2 * f5 + 2 * f6 +
            f7 + f8 + f9 - m18eq;
  m19diff = f10 - f11 - f12 - f13 - f14 - 2 * f3 - 2 * f4 + 2 * f5 + 2 * f6 +
            f7 + f8 + f9 - m19eq;
  m20diff = -2 * f10 + f19 - f20 - f21 + f22 + f23 - f24 - f25 + f26 - 2 * f7 +
            2 * f8 + 2 * f9 - m20eq;
  m21diff = -2 * f15 + 2 * f16 + 2 * f17 - 2 * f18 + f19 + f20 - f21 - f22 -
            f23 - f24 + f25 + f26 - m21eq;
  m22diff = -2 * f11 + 2 * f12 + 2 * f13 - 2 * f14 + f19 - f20 + f21 - f22 -
            f23 + f24 - f25 + f26 - m22eq;
  m23diff = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9 - m23eq;
  m24diff = f10 + f15 - f16 + f17 - f18 - f7 - f8 + f9 - m24eq;
  m25diff = f11 + f12 - f13 - f14 - f15 - f16 + f17 + f18 - m25eq;
  m26diff = f19 - f20 - f21 + f22 - f23 + f24 + f25 - f26 - m26eq;

  // Non equilibrium stress-tensor for velocity
  Sxx = m10diff + m11diff + m12diff + m13diff + m14diff + m19diff + m1diff +
        m20diff + m21diff + m22diff + m23diff + m24diff + m25diff + m26diff +
        m2diff + m7diff + m8diff + m9diff;
  Syy = m10diff + m15diff + m16diff + m17diff + m18diff + m19diff + m20diff +
        m21diff + m22diff + m23diff + m24diff + m25diff + m26diff + m3diff +
        m4diff + m7diff + m8diff + m9diff;
  Szz = m11diff + m12diff + m13diff + m14diff + m15diff + m16diff + m17diff +
        m18diff + m19diff + m20diff + m21diff + m22diff + m23diff + m24diff +
        m25diff + m26diff + m5diff + m6diff;
  Sxy = m10diff + m19diff - m20diff - m21diff + m22diff + m23diff - m24diff -
        m25diff + m26diff + m7diff - m8diff - m9diff;
  Sxz = m11diff - m12diff - m13diff + m14diff + m19diff - m20diff + m21diff -
        m22diff - m23diff + m24diff - m25diff + m26diff;
  Syz = m15diff - m16diff - m17diff + m18diff + m19diff + m20diff - m21diff -
        m22diff - m23diff - m24diff + m25diff + m26diff;

  // Magnitude of strain rate tensor
  S_bar = 1.4142135623730951f *
          powf(0.5f * (Sxx) * (Sxx) + (Sxy) * (Sxy) + (Sxz) * (Sxz) +
                   0.5f * (Syy) * (Syy) + (Syz) * (Syz) + 0.5f * (Szz) * (Szz),
               0.5f);
  ST = -0.16666666666666666f * nu +
       0.70710678118654746f *
           powf((C) * (C)*S_bar + 0.055555555555555552f * (nu) * (nu), 0.5f);

  // Modified shear viscosity
  tau_V = 1.0f / (3.0f * ST + 3.0f * nu + 0.5f);

  // Relax velocity
  omega0 = 0.15555555555555553f * m16diff - 0.05962962962962963f * m17diff -
           0.16666666666666666f * m4diff;
  omega1 = -0.083333333333333329f * m10diff + 0.10166666666666667f * m13diff +
           0.029814814814814815f * m17diff - 0.10999999999999999f * m18diff -
           0.083333333333333329f * m4diff + (1.0f / 18.0f) * m5diff * tau_V;
  omega2 = 0.083333333333333329f * m10diff - 0.10166666666666667f * m13diff +
           0.029814814814814815f * m17diff - 0.10999999999999999f * m18diff -
           0.083333333333333329f * m4diff + (1.0f / 18.0f) * m5diff * tau_V;
  omega3 = -0.083333333333333329f * m11diff + 0.10166666666666667f * m14diff +
           0.029814814814814815f * m17diff + 0.054999999999999993f * m18diff -
           0.16499999999999998f * m19diff - 0.083333333333333329f * m4diff -
           1.0f / 36.0f * m5diff * tau_V + (1.0f / 12.0f) * m6diff * tau_V;
  omega4 = 0.083333333333333329f * m11diff - 0.10166666666666667f * m14diff +
           0.029814814814814815f * m17diff + 0.054999999999999993f * m18diff -
           0.16499999999999998f * m19diff - 0.083333333333333329f * m4diff -
           1.0f / 36.0f * m5diff * tau_V + (1.0f / 12.0f) * m6diff * tau_V;
  omega5 = -0.083333333333333329f * m12diff + 0.10166666666666667f * m15diff +
           0.029814814814814815f * m17diff + 0.054999999999999993f * m18diff +
           0.16499999999999998f * m19diff - 0.083333333333333329f * m4diff -
           1.0f / 36.0f * m5diff * tau_V - 1.0f / 12.0f * m6diff * tau_V;
  omega6 = 0.083333333333333329f * m12diff - 0.10166666666666667f * m15diff +
           0.029814814814814815f * m17diff + 0.054999999999999993f * m18diff +
           0.16499999999999998f * m19diff - 0.083333333333333329f * m4diff -
           1.0f / 36.0f * m5diff * tau_V - 1.0f / 12.0f * m6diff * tau_V;
  omega7 = -0.020833333333333332f * m10diff - 0.020833333333333332f * m11diff -
           0.050833333333333335f * m13diff - 0.050833333333333335f * m14diff -
           0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
           0.027499999999999997f * m18diff + 0.08249999999999999f * m19diff -
           0.16499999999999998f * m20diff + 0.2175f * m23diff -
           0.2175f * m24diff + (1.0f / 36.0f) * m5diff * tau_V +
           (1.0f / 12.0f) * m6diff * tau_V + (1.0f / 12.0f) * m7diff * tau_V;
  omega8 = 0.020833333333333332f * m10diff - 0.020833333333333332f * m11diff +
           0.050833333333333335f * m13diff - 0.050833333333333335f * m14diff -
           0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
           0.027499999999999997f * m18diff + 0.08249999999999999f * m19diff +
           0.16499999999999998f * m20diff - 0.2175f * m23diff -
           0.2175f * m24diff + (1.0f / 36.0f) * m5diff * tau_V +
           (1.0f / 12.0f) * m6diff * tau_V - 1.0f / 12.0f * m7diff * tau_V;
  omega9 = -0.020833333333333332f * m10diff + 0.020833333333333332f * m11diff -
           0.050833333333333335f * m13diff + 0.050833333333333335f * m14diff -
           0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
           0.027499999999999997f * m18diff + 0.08249999999999999f * m19diff +
           0.16499999999999998f * m20diff + 0.2175f * m23diff +
           0.2175f * m24diff + (1.0f / 36.0f) * m5diff * tau_V +
           (1.0f / 12.0f) * m6diff * tau_V - 1.0f / 12.0f * m7diff * tau_V;
  omega10 = 0.020833333333333332f * m10diff + 0.020833333333333332f * m11diff +
            0.050833333333333335f * m13diff + 0.050833333333333335f * m14diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
            0.027499999999999997f * m18diff + 0.08249999999999999f * m19diff -
            0.16499999999999998f * m20diff - 0.2175f * m23diff +
            0.2175f * m24diff + (1.0f / 36.0f) * m5diff * tau_V +
            (1.0f / 12.0f) * m6diff * tau_V + (1.0f / 12.0f) * m7diff * tau_V;
  omega11 = -0.020833333333333332f * m10diff - 0.020833333333333332f * m12diff -
            0.050833333333333335f * m13diff - 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
            0.027499999999999997f * m18diff - 0.08249999999999999f * m19diff -
            0.16499999999999998f * m22diff - 0.2175f * m23diff +
            0.2175f * m25diff + (1.0f / 36.0f) * m5diff * tau_V -
            1.0f / 12.0f * m6diff * tau_V + (1.0f / 12.0f) * m9diff * tau_V;
  omega12 = 0.020833333333333332f * m10diff - 0.020833333333333332f * m12diff +
            0.050833333333333335f * m13diff - 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
            0.027499999999999997f * m18diff - 0.08249999999999999f * m19diff +
            0.16499999999999998f * m22diff + 0.2175f * m23diff +
            0.2175f * m25diff + (1.0f / 36.0f) * m5diff * tau_V -
            1.0f / 12.0f * m6diff * tau_V - 1.0f / 12.0f * m9diff * tau_V;
  omega13 = -0.020833333333333332f * m10diff + 0.020833333333333332f * m12diff -
            0.050833333333333335f * m13diff + 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
            0.027499999999999997f * m18diff - 0.08249999999999999f * m19diff +
            0.16499999999999998f * m22diff - 0.2175f * m23diff -
            0.2175f * m25diff + (1.0f / 36.0f) * m5diff * tau_V -
            1.0f / 12.0f * m6diff * tau_V - 1.0f / 12.0f * m9diff * tau_V;
  omega14 = 0.020833333333333332f * m10diff + 0.020833333333333332f * m12diff +
            0.050833333333333335f * m13diff + 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff +
            0.027499999999999997f * m18diff - 0.08249999999999999f * m19diff -
            0.16499999999999998f * m22diff + 0.2175f * m23diff -
            0.2175f * m25diff + (1.0f / 36.0f) * m5diff * tau_V -
            1.0f / 12.0f * m6diff * tau_V + (1.0f / 12.0f) * m9diff * tau_V;
  omega15 = -0.020833333333333332f * m11diff - 0.020833333333333332f * m12diff -
            0.050833333333333335f * m14diff - 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff -
            0.054999999999999993f * m18diff - 0.16499999999999998f * m21diff +
            0.2175f * m24diff - 0.2175f * m25diff -
            1.0f / 18.0f * m5diff * tau_V + (1.0f / 12.0f) * m8diff * tau_V;
  omega16 = 0.020833333333333332f * m11diff - 0.020833333333333332f * m12diff +
            0.050833333333333335f * m14diff - 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff -
            0.054999999999999993f * m18diff + 0.16499999999999998f * m21diff -
            0.2175f * m24diff - 0.2175f * m25diff -
            1.0f / 18.0f * m5diff * tau_V - 1.0f / 12.0f * m8diff * tau_V;
  omega17 = -0.020833333333333332f * m11diff + 0.020833333333333332f * m12diff -
            0.050833333333333335f * m14diff + 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff -
            0.054999999999999993f * m18diff + 0.16499999999999998f * m21diff +
            0.2175f * m24diff + 0.2175f * m25diff -
            1.0f / 18.0f * m5diff * tau_V - 1.0f / 12.0f * m8diff * tau_V;
  omega18 = 0.020833333333333332f * m11diff + 0.020833333333333332f * m12diff +
            0.050833333333333335f * m14diff + 0.050833333333333335f * m15diff -
            0.038888888888888883f * m16diff - 0.014907407407407407f * m17diff -
            0.054999999999999993f * m18diff - 0.16499999999999998f * m21diff -
            0.2175f * m24diff + 0.2175f * m25diff -
            1.0f / 18.0f * m5diff * tau_V + (1.0f / 12.0f) * m8diff * tau_V;
  omega19 = 0.041666666666666664f * m10diff + 0.041666666666666664f * m11diff +
            0.041666666666666664f * m12diff + 0.025416666666666667f * m13diff +
            0.025416666666666667f * m14diff + 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff +
            0.08249999999999999f * m20diff + 0.08249999999999999f * m21diff +
            0.08249999999999999f * m22diff + 0.2175f * m26diff +
            0.083333333333333329f * m4diff + (1.0f / 12.0f) * m7diff * tau_V +
            (1.0f / 12.0f) * m8diff * tau_V + (1.0f / 12.0f) * m9diff * tau_V;
  omega20 = -0.041666666666666664f * m10diff + 0.041666666666666664f * m11diff +
            0.041666666666666664f * m12diff - 0.025416666666666667f * m13diff +
            0.025416666666666667f * m14diff + 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff -
            0.08249999999999999f * m20diff + 0.08249999999999999f * m21diff -
            0.08249999999999999f * m22diff - 0.2175f * m26diff +
            0.083333333333333329f * m4diff - 1.0f / 12.0f * m7diff * tau_V +
            (1.0f / 12.0f) * m8diff * tau_V - 1.0f / 12.0f * m9diff * tau_V;
  omega21 = 0.041666666666666664f * m10diff - 0.041666666666666664f * m11diff +
            0.041666666666666664f * m12diff + 0.025416666666666667f * m13diff -
            0.025416666666666667f * m14diff + 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff -
            0.08249999999999999f * m20diff - 0.08249999999999999f * m21diff +
            0.08249999999999999f * m22diff - 0.2175f * m26diff +
            0.083333333333333329f * m4diff - 1.0f / 12.0f * m7diff * tau_V -
            1.0f / 12.0f * m8diff * tau_V + (1.0f / 12.0f) * m9diff * tau_V;
  omega22 = -0.041666666666666664f * m10diff - 0.041666666666666664f * m11diff +
            0.041666666666666664f * m12diff - 0.025416666666666667f * m13diff -
            0.025416666666666667f * m14diff + 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff +
            0.08249999999999999f * m20diff - 0.08249999999999999f * m21diff -
            0.08249999999999999f * m22diff + 0.2175f * m26diff +
            0.083333333333333329f * m4diff + (1.0f / 12.0f) * m7diff * tau_V -
            1.0f / 12.0f * m8diff * tau_V - 1.0f / 12.0f * m9diff * tau_V;
  omega23 = 0.041666666666666664f * m10diff + 0.041666666666666664f * m11diff -
            0.041666666666666664f * m12diff + 0.025416666666666667f * m13diff +
            0.025416666666666667f * m14diff - 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff +
            0.08249999999999999f * m20diff - 0.08249999999999999f * m21diff -
            0.08249999999999999f * m22diff - 0.2175f * m26diff +
            0.083333333333333329f * m4diff + (1.0f / 12.0f) * m7diff * tau_V -
            1.0f / 12.0f * m8diff * tau_V - 1.0f / 12.0f * m9diff * tau_V;
  omega24 = -0.041666666666666664f * m10diff + 0.041666666666666664f * m11diff -
            0.041666666666666664f * m12diff - 0.025416666666666667f * m13diff +
            0.025416666666666667f * m14diff - 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff -
            0.08249999999999999f * m20diff - 0.08249999999999999f * m21diff +
            0.08249999999999999f * m22diff + 0.2175f * m26diff +
            0.083333333333333329f * m4diff - 1.0f / 12.0f * m7diff * tau_V -
            1.0f / 12.0f * m8diff * tau_V + (1.0f / 12.0f) * m9diff * tau_V;
  omega25 = 0.041666666666666664f * m10diff - 0.041666666666666664f * m11diff -
            0.041666666666666664f * m12diff + 0.025416666666666667f * m13diff -
            0.025416666666666667f * m14diff - 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff -
            0.08249999999999999f * m20diff + 0.08249999999999999f * m21diff -
            0.08249999999999999f * m22diff + 0.2175f * m26diff +
            0.083333333333333329f * m4diff - 1.0f / 12.0f * m7diff * tau_V +
            (1.0f / 12.0f) * m8diff * tau_V - 1.0f / 12.0f * m9diff * tau_V;
  omega26 = -0.041666666666666664f * m10diff - 0.041666666666666664f * m11diff -
            0.041666666666666664f * m12diff - 0.025416666666666667f * m13diff -
            0.025416666666666667f * m14diff - 0.025416666666666667f * m15diff +
            0.038888888888888883f * m16diff + 0.0074537037037037037f * m17diff +
            0.08249999999999999f * m20diff + 0.08249999999999999f * m21diff +
            0.08249999999999999f * m22diff - 0.2175f * m26diff +
            0.083333333333333329f * m4diff + (1.0f / 12.0f) * m7diff * tau_V +
            (1.0f / 12.0f) * m8diff * tau_V + (1.0f / 12.0f) * m9diff * tau_V;

  // Macroscopic temperature
  T = T0 + T1 + T2 + T3 + T4 + T5 + T6;

  // Temperature equilibirum distribution functions
  T0eq = 0.14285714285714285f * T;
  T1eq = 0.14285714285714285f * T * (3.5f * ux + 1.0f);
  T2eq = 0.14285714285714285f * T * (1.0f - 3.5f * ux);
  T3eq = 0.14285714285714285f * T * (3.5f * uy + 1.0f);
  T4eq = 0.14285714285714285f * T * (1.0f - 3.5f * uy);
  T5eq = 0.14285714285714285f * T * (3.5f * uz + 1.0f);
  T6eq = 0.14285714285714285f * T * (1.0f - 3.5f * uz);

  // Modified relaxation time for the temperature
  tau_T = 3.0f * nuT + 0.5f + 3.0f * ST / Pr_t;

  // Boussinesq approximation of body force
  Fup = gBetta * (T - Tref);
  Fdown = -gBetta * (T - Tref);

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
  dftmp3D(19, x, y, z, nx, ny, nz) = f19 - omega19;
  dftmp3D(20, x, y, z, nx, ny, nz) = f20 - omega20;
  dftmp3D(21, x, y, z, nx, ny, nz) = f21 - omega21;
  dftmp3D(22, x, y, z, nx, ny, nz) = f22 - omega22;
  dftmp3D(23, x, y, z, nx, ny, nz) = f23 - omega23;
  dftmp3D(24, x, y, z, nx, ny, nz) = f24 - omega24;
  dftmp3D(25, x, y, z, nx, ny, nz) = f25 - omega25;
  dftmp3D(26, x, y, z, nx, ny, nz) = f26 - omega26;
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
  phy->vx = ux;
  phy->vy = uy;
  phy->vz = uz;
}
