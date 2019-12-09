#include "Kernel.hpp"

__device__ PhysicalQuantity compute(
    // Lattice position in partition
    const Eigen::Vector3i pos,
    // Size of partition
    const Eigen::Vector3i size,
    // Size of ghostLayer
    const Eigen::Vector3i ghostLayer,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
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
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst) {
  const voxel_t voxelID = voxels[I3D(pos, size)];
  if (voxelID == -1) return {.rho = 0, .T = 0, .vx = 0, .vy = 0, .vz = 0};

  const BoundaryCondition bc = bcs[voxelID];

  // Calculate array position for distribution functions (with ghostLayers)
  const int nx = size.x() + ghostLayer.x() * 2;
  const int ny = size.y() + ghostLayer.y() * 2;
  const int nz = size.z() + ghostLayer.z() * 2;

  const int x = pos.x() + ghostLayer.x();
  const int y = pos.y() + ghostLayer.y();
  const int z = pos.z() + ghostLayer.z();

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

  if (bc.m_type == VoxelType::WALL) {
    // Half-way bounceback
    const real3 v =
        make_float3(bc.m_velocity.x(), bc.m_velocity.y(), bc.m_velocity.z());
    const real3 n =
        make_float3(bc.m_normal.x(), bc.m_normal.y(), bc.m_normal.z());
// BC for velocity dfs
#pragma unroll
    for (int i = 0; i < 19; i++) {
      const real3 ei =
          make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                      D3Q27directions[i * 3 + 2]);
      if (dot(ei, n) > 0.0)
        *fs[i] = df3D(D3Q27directionsOpposite[i], x, y, z, nx, ny, nz);
    }
// BC for temperature dfs
#pragma unroll
    for (int i = 1; i < 7; i++) {
      const real3 ei =
          make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                      D3Q27directions[i * 3 + 2]);
      if (dot(ei, n) > 0.0)
        *Ts[i] = Tdf3D(D3Q27directionsOpposite[i], x, y, z, nx, ny, nz);
    }
    /////////////////////////////
  } else if (bc.m_type == VoxelType::INLET_CONSTANT ||
             bc.m_type == VoxelType::INLET_RELATIVE ||
             bc.m_type == VoxelType::INLET_ZERO_GRADIENT) {
    // Inlet boundary condition
    const real3 v =
        make_float3(bc.m_velocity.x(), bc.m_velocity.y(), bc.m_velocity.z());
    const real3 n =
        make_float3(bc.m_normal.x(), bc.m_normal.y(), bc.m_normal.z());
// BC for velocity dfs
#pragma unroll
    for (int i = 0; i < 19; i++) {
      const real3 ei =
          make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                      D3Q27directions[i * 3 + 2]);
      const real dot_vv = dot(v, v);
      if (dot(ei, n) > 0.0) {
        const real wi = D3Q19weights[i];
        const real rho = 1.0;
        const real dot_eiv = dot(ei, v);
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
      const real3 ei =
          make_float3(D3Q27directions[i * 3], D3Q27directions[i * 3 + 1],
                      D3Q27directions[i * 3 + 2]);
      const real wi = D3Q7weights[i];
      if (dot(ei, n) > 0.0) {
        if (bc.m_type == VoxelType::INLET_CONSTANT) {
          *Ts[i] = real(wi * bc.m_temperature * (1.0 + 3.0 * dot(ei, v)));
        } else if (bc.m_type == VoxelType::INLET_ZERO_GRADIENT) {
          // approximate a first order expansion
          *Ts[i] = Tdf3D(i, x + bc.m_normal.x(), y + bc.m_normal.y(),
                         z + bc.m_normal.z(), nx, ny, nz);
        } else if (bc.m_type == VoxelType::INLET_RELATIVE) {
          // compute macroscopic temperature at the relative position
          real Trel = 0;
#pragma unroll
          for (int qIdx = 1; qIdx < 7; qIdx++)
            Trel =
                Trel + Tdf3D(qIdx, x + bc.m_rel_pos.x(), y + bc.m_rel_pos.y(),
                             z + bc.m_rel_pos.z(), nx, ny, nz);
          *Ts[i] =
              real((Trel + bc.m_temperature) * (wi * (1.0 + 3.0 * dot(ei, v))));
        }
      }
    }
  }

  real m0eq, m1eq, m2eq, m3eq, m4eq, m5eq, m6eq, m7eq, m8eq, m9eq, m10eq, m11eq,
      m12eq, m13eq, m14eq, m15eq, m16eq, m17eq, m18eq;
  real m0neq, m1neq, m2neq, m3neq, m4neq, m5neq, m6neq, m7neq, m8neq, m9neq,
      m10neq, m11neq, m12neq, m13neq, m14neq, m15neq, m16neq, m17neq, m18neq;
  real omega0, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8,
      omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16,
      omega17, omega18;
  real n0eq, n1eq, n2eq, n3eq, n4eq, n5eq, n6eq;
  real n0neq, n1neq, n2neq, n3neq, n4neq, n5neq, n6neq;
  real omegaT0, omegaT1, omegaT2, omegaT3, omegaT4, omegaT5, omegaT6;
  real S_bar;
  real nu_t;
  real T;
  real Fup;
  real Fdown;
  real Sxx, Syy, Szz, Sxy, Syz, Sxz;
  real m1_1, m1_9, m1_11, m1_13, m1_14, m1_15;
  real rho, en, epsilon, jx, qx, jy, qy, jz, qz, pxx3, pixx3, pww, piww, pxy,
      pyz, pxz, mx, my, mz;
  real omega_e, omega_xx, omega_ej;
  real vx, vy, vz;

  // Macroscopic density
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
        f3 + f4 + f5 + f6 + f7 + f8 + f9;

  // Transform velocity distribution functions to moment space
  en = -30.0 * f0 - 11.0 * f1 + 8.0 * f10 + 8.0 * f11 + 8.0 * f12 + 8.0 * f13 +
       8.0 * f14 + 8.0 * f15 + 8.0 * f16 + 8.0 * f17 + 8.0 * f18 - 11.0 * f2 -
       11.0 * f3 - 11.0 * f4 - 11.0 * f5 - 11.0 * f6 + 8.0 * f7 + 8.0 * f8 +
       8.0 * f9;
  epsilon = 12.0 * f0 - 4.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 +
            1.0 * f13 + 1.0 * f14 + 1.0 * f15 + 1.0 * f16 + 1.0 * f17 +
            1.0 * f18 - 4.0 * f2 - 4.0 * f3 - 4.0 * f4 - 4.0 * f5 - 4.0 * f6 +
            1.0 * f7 + 1.0 * f8 + 1.0 * f9;
  jx = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
  jy = f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9;
  jz = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6;
  qx = -4.0 * f1 - 1.0 * f10 + 1.0 * f11 - 1.0 * f12 + 1.0 * f13 - 1.0 * f14 +
       4.0 * f2 + 1.0 * f7 - 1.0 * f8 + 1.0 * f9;
  qy = 1.0 * f10 + 1.0 * f15 - 1.0 * f16 + 1.0 * f17 - 1.0 * f18 - 4.0 * f3 +
       4.0 * f4 + 1.0 * f7 - 1.0 * f8 - 1.0 * f9;
  qz = 1.0 * f11 - 1.0 * f12 - 1.0 * f13 + 1.0 * f14 + 1.0 * f15 - 1.0 * f16 -
       1.0 * f17 + 1.0 * f18 - 4.0 * f5 + 4.0 * f6;
  pxx3 = 2.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 + 1.0 * f13 + 1.0 * f14 -
         2 * f15 - 2 * f16 - 2 * f17 - 2 * f18 + 2.0 * f2 - f3 - f4 - f5 - f6 +
         1.0 * f7 + 1.0 * f8 + 1.0 * f9;
  pixx3 = -4.0 * f1 + 1.0 * f10 + 1.0 * f11 + 1.0 * f12 + 1.0 * f13 +
          1.0 * f14 - 2.0 * f15 - 2.0 * f16 - 2.0 * f17 - 2.0 * f18 - 4.0 * f2 +
          2.0 * f3 + 2.0 * f4 + 2.0 * f5 + 2.0 * f6 + 1.0 * f7 + 1.0 * f8 +
          1.0 * f9;
  pww = f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9;
  piww = 1.0 * f10 - 1.0 * f11 - 1.0 * f12 - 1.0 * f13 - 1.0 * f14 - 2.0 * f3 -
         2.0 * f4 + 2.0 * f5 + 2.0 * f6 + 1.0 * f7 + 1.0 * f8 + 1.0 * f9;
  pxy = -f10 + f7 + f8 - f9;
  pyz = f15 + f16 - f17 - f18;
  pxz = f11 + f12 - f13 - f14;
  mx = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9;
  my = -f10 + f15 - f16 + f17 - f18 - f7 + f8 + f9;
  mz = f11 - f12 - f13 + f14 - f15 + f16 + f17 - f18;
  omega_e = 0;
  omega_xx = 0;
  omega_ej = -7.53968253968254;

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
  m4eq = -0.666666666666667 * jx;
  m5eq = jy;
  m6eq = -0.666666666666667 * jy;
  m7eq = jz;
  m8eq = -0.666666666666667 * jz;
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

  // LES strain rate tensor
  m1_1 = 12.6666666666667 * jx + 12.6666666666667 * jy + 12.6666666666667 * jz;
  m1_9 =
      -1.33333333333333 * jx + 0.666666666666667 * jy + 0.666666666666667 * jz;
  m1_11 = -0.666666666666667 * jy + 0.666666666666667 * jz;
  m1_13 = -0.333333333333333 * jx - 0.333333333333333 * jy;
  m1_14 = -0.333333333333333 * jy - 0.333333333333333 * jz;
  m1_15 = -0.333333333333333 * jx - 0.333333333333333 * jz;
  Sxx = -0.0263157894736842 * m1_1 - 0.5 * m1_9;
  Syy = -0.0263157894736842 * m1_1 - 0.75 * m1_11 + 0.25 * m1_9;
  Szz = -0.0263157894736842 * m1_1 + 0.75 * m1_11 + 0.25 * m1_9;
  Sxy = -1.5 * m1_13;
  Syz = -1.5 * m1_14;
  Sxz = -1.5 * m1_15;
  S_bar = sqrt(2.0 * pow(Sxx, 2) + 2.0 * pow(Sxy, 2) + 2.0 * pow(Sxz, 2) +
               2.0 * pow(Syy, 2) + 2.0 * pow(Syz, 2) + 2.0 * pow(Szz, 2));

  // Eddy viscosity
  nu_t = 4.0 * pow(C, 2) * S_bar;

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
      0.0555555555555556 * gBetta * rho * (T - Tref) * (-vz + 1) *
      (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) + 3.0 * pow(vz, 2) + 3.0 * vz + 1);
  Fdown =
      0.0555555555555556 * gBetta * rho * (T - Tref) * (-vz - 1) *
      (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) + 3.0 * pow(vz, 2) - 3.0 * vz + 1);

  // Difference to velocity equilibrium
  m0neq = -m0eq + rho;
  m1neq = en - m1eq;
  m2neq = epsilon - m2eq;
  m3neq = jx - m3eq;
  m4neq = -m4eq + qx;
  m5neq = jy - m5eq;
  m6neq = -m6eq + qy;
  m7neq = jz - m7eq;
  m8neq = -m8eq + qz;
  m9neq = -m9eq + pxx3;
  m10neq = -m10eq + pixx3;
  m11neq = -m11eq + pww;
  m12neq = -m12eq + piww;
  m13neq = -m13eq + pxy;
  m14neq = -m14eq + pyz;
  m15neq = -m15eq + pxz;
  m16neq = -m16eq + mx;
  m17neq = -m17eq + my;
  m18neq = -m18eq + mz;

  // Relax velocity
  omega0 = -0.0149122807017544 * m1neq + 0.0666666666666667 * m2neq;
  omega1 = -0.0777777777777778 * m10neq - 0.00546783625730994 * m1neq -
           0.0222222222222222 * m2neq - 0.12 * m4neq +
           0.111111111111111 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega2 = -0.0777777777777778 * m10neq - 0.00546783625730994 * m1neq -
           0.0222222222222222 * m2neq + 0.12 * m4neq +
           0.111111111111111 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega3 = 0.0388888888888889 * m10neq +
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) -
           0.116666666666667 * m12neq - 0.00546783625730994 * m1neq -
           0.0222222222222222 * m2neq - 0.12 * m6neq -
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega4 = 0.0388888888888889 * m10neq +
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) -
           0.116666666666667 * m12neq - 0.00546783625730994 * m1neq -
           0.0222222222222222 * m2neq + 0.12 * m6neq -
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega5 = 0.0388888888888889 * m10neq -
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) +
           0.116666666666667 * m12neq - 0.00546783625730994 * m1neq -
           0.0222222222222222 * m2neq - 0.12 * m8neq -
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega6 = 0.0388888888888889 * m10neq -
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) +
           0.116666666666667 * m12neq - 0.00546783625730994 * m1neq -
           0.0222222222222222 * m2neq + 0.12 * m8neq -
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega7 = 0.0194444444444444 * m10neq +
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) +
           0.0583333333333333 * m12neq +
           0.5 * m13neq / (6.0 * nu + 6.0 * nu_t + 1.0) + 0.2475 * m16neq -
           0.2475 * m17neq + 0.0039766081871345 * m1neq +
           0.00555555555555555 * m2neq + 0.03 * m4neq + 0.03 * m6neq +
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega8 = 0.0194444444444444 * m10neq +
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) +
           0.0583333333333333 * m12neq +
           0.5 * m13neq / (6.0 * nu + 6.0 * nu_t + 1.0) - 0.2475 * m16neq +
           0.2475 * m17neq + 0.0039766081871345 * m1neq +
           0.00555555555555555 * m2neq - 0.03 * m4neq - 0.03 * m6neq +
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega9 = 0.0194444444444444 * m10neq +
           0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) +
           0.0583333333333333 * m12neq -
           0.5 * m13neq / (6.0 * nu + 6.0 * nu_t + 1.0) + 0.2475 * m16neq +
           0.2475 * m17neq + 0.0039766081871345 * m1neq +
           0.00555555555555555 * m2neq + 0.03 * m4neq - 0.03 * m6neq +
           0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega10 = 0.0194444444444444 * m10neq +
            0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) +
            0.0583333333333333 * m12neq -
            0.5 * m13neq / (6.0 * nu + 6.0 * nu_t + 1.0) - 0.2475 * m16neq -
            0.2475 * m17neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq - 0.03 * m4neq + 0.03 * m6neq +
            0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega11 = 0.0194444444444444 * m10neq -
            0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) -
            0.0583333333333333 * m12neq +
            0.5 * m15neq / (6.0 * nu + 6.0 * nu_t + 1.0) - 0.2475 * m16neq +
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq + 0.03 * m4neq + 0.03 * m8neq +
            0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega12 = 0.0194444444444444 * m10neq -
            0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) -
            0.0583333333333333 * m12neq +
            0.5 * m15neq / (6.0 * nu + 6.0 * nu_t + 1.0) + 0.2475 * m16neq -
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq - 0.03 * m4neq - 0.03 * m8neq +
            0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega13 = 0.0194444444444444 * m10neq -
            0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) -
            0.0583333333333333 * m12neq -
            0.5 * m15neq / (6.0 * nu + 6.0 * nu_t + 1.0) - 0.2475 * m16neq -
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq + 0.03 * m4neq - 0.03 * m8neq +
            0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega14 = 0.0194444444444444 * m10neq -
            0.166666666666667 * m11neq / (6.0 * nu + 6.0 * nu_t + 1.0) -
            0.0583333333333333 * m12neq -
            0.5 * m15neq / (6.0 * nu + 6.0 * nu_t + 1.0) + 0.2475 * m16neq +
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq - 0.03 * m4neq + 0.03 * m8neq +
            0.0555555555555556 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega15 = -0.0388888888888889 * m10neq +
            0.5 * m14neq / (6.0 * nu + 6.0 * nu_t + 1.0) + 0.2475 * m17neq -
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq + 0.03 * m6neq + 0.03 * m8neq -
            0.111111111111111 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega16 = -0.0388888888888889 * m10neq +
            0.5 * m14neq / (6.0 * nu + 6.0 * nu_t + 1.0) - 0.2475 * m17neq +
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq - 0.03 * m6neq - 0.03 * m8neq -
            0.111111111111111 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega17 = -0.0388888888888889 * m10neq -
            0.5 * m14neq / (6.0 * nu + 6.0 * nu_t + 1.0) + 0.2475 * m17neq +
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq + 0.03 * m6neq - 0.03 * m8neq -
            0.111111111111111 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);
  omega18 = -0.0388888888888889 * m10neq -
            0.5 * m14neq / (6.0 * nu + 6.0 * nu_t + 1.0) - 0.2475 * m17neq -
            0.2475 * m18neq + 0.0039766081871345 * m1neq +
            0.00555555555555555 * m2neq - 0.03 * m6neq + 0.03 * m8neq -
            0.111111111111111 * m9neq / (6.0 * nu + 6.0 * nu_t + 1.0);

  // Difference to temperature equilibrium
  n0neq = T0 + T1 + T2 + T3 + T4 + T5 + T6 - n0eq;
  n1neq = T1 - T2 - n1eq;
  n2neq = T3 - T4 - n2eq;
  n3neq = T5 - T6 - n3eq;
  n4neq = 6.0 * T0 - 1.0 * T1 - 1.0 * T2 - 1.0 * T3 - 1.0 * T4 - 1.0 * T5 -
          1.0 * T6 - n4eq;
  n5neq = 2.0 * T1 + 2.0 * T2 - T3 - T4 - T5 - T6 - n5eq;
  n6neq = T3 + T4 - T5 - T6 - n6eq;

  // Relax temperature
  omegaT0 = 0.142857142857143 * n0neq + 0.142857142857143 * n4neq;
  omegaT1 = 0.142857142857143 * n0neq +
            0.5 * n1neq * (4.0 * nuT + 0.5 + 4.0 * nu_t / Pr_t) -
            0.0238095238095238 * n4neq + 0.166666666666667 * n5neq;
  omegaT2 = 0.142857142857143 * n0neq -
            0.5 * n1neq * (4.0 * nuT + 0.5 + 4.0 * nu_t / Pr_t) -
            0.0238095238095238 * n4neq + 0.166666666666667 * n5neq;
  omegaT3 = 0.142857142857143 * n0neq +
            0.5 * n2neq * (4.0 * nuT + 0.5 + 4.0 * nu_t / Pr_t) -
            0.0238095238095238 * n4neq - 0.0833333333333333 * n5neq +
            0.25 * n6neq;
  omegaT4 = 0.142857142857143 * n0neq -
            0.5 * n2neq * (4.0 * nuT + 0.5 + 4.0 * nu_t / Pr_t) -
            0.0238095238095238 * n4neq - 0.0833333333333333 * n5neq +
            0.25 * n6neq;
  omegaT5 = 0.142857142857143 * n0neq +
            0.5 * n3neq * (4.0 * nuT + 0.5 + 4.0 * nu_t / Pr_t) -
            0.0238095238095238 * n4neq - 0.0833333333333333 * n5neq -
            0.25 * n6neq;
  omegaT6 = 0.142857142857143 * n0neq -
            0.5 * n3neq * (4.0 * nuT + 0.5 + 4.0 * nu_t / Pr_t) -
            0.0238095238095238 * n4neq - 0.0833333333333333 * n5neq -
            0.25 * n6neq;

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

  const PhysicalQuantity phy = {
      .rho = rho, .T = T, .vx = vx, .vy = vy, .vz = vz};

  // Average temperature and velocity
  averageDst[I4D(0, pos, size)] = averageSrc[I4D(0, pos, size)] + phy.T;
  averageDst[I4D(1, pos, size)] = averageSrc[I4D(1, pos, size)] + phy.vx;
  averageDst[I4D(2, pos, size)] = averageSrc[I4D(2, pos, size)] + phy.vy;
  averageDst[I4D(3, pos, size)] = averageSrc[I4D(3, pos, size)] + phy.vz;

  return phy;
}

__device__ void computeAndPlot(
    // Lattice position in partition
    const Eigen::Vector3i pos,
    // Size of partition
    const Eigen::Vector3i size,
    // Size of ghostLayer
    const Eigen::Vector3i ghostLayer,
    // Velocity distribution functions
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    // Temperature distribution functions
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    // Voxel type array
    const voxel_t* __restrict__ voxels,
    // Boundary condition data
    BoundaryCondition* __restrict__ bcs,
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
    // Contain the macroscopic temperature, velocity (x,y,z components)
    //  integrated in time (so /nbr_of_time_steps to get average)
    real* __restrict__ averageSrc,
    real* __restrict__ averageDst,
    // Quantity to be visualised
    const DisplayQuantity::Enum displayQuantity,
    // Plot array for display
    real* __restrict__ plot) {
  // Type of voxel for calculating boundary conditions
  const voxel_t voxelID = voxels[I3D(pos, size)];

  // Plot empty voxels
  if (voxelID == -1) {
    plot[I3D(pos, size)] = REAL_NAN;
    return;
  }

  PhysicalQuantity phy =
      compute(pos, size, ghostLayer, df, df_tmp, dfT, dfT_tmp, voxels, bcs, nu,
              C, nuT, Pr_t, gBetta, Tref, averageSrc, averageDst);

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

__global__ void ComputeKernelInterior(const Partition partition,
                                      real* __restrict__ df,
                                      real* __restrict__ df_tmp,
                                      real* __restrict__ dfT,
                                      real* __restrict__ dfT_tmp,
                                      const voxel_t* __restrict__ voxels,
                                      BoundaryCondition* __restrict__ bcs,
                                      const real nu,
                                      const real C,
                                      const real nuT,
                                      const real Pr_t,
                                      const real gBetta,
                                      const real Tref,
                                      real* __restrict__ averageSrc,
                                      real* __restrict__ averageDst) {
  Eigen::Vector3i partSize = partition.getExtents();
  Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x + partGhostLayer.x();
  const int y = blockIdx.x + partGhostLayer.y();
  const int z = blockIdx.y + partGhostLayer.z();

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeKernelBoundaryX(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = blockIdx.y * (partSize.x() - 1);  // Might not be multiple of 32
  const int y = threadIdx.x;
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeKernelBoundaryY(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.y * (partSize.y() - 1);
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeKernelBoundaryZ(const Partition partition,
                                       real* __restrict__ df,
                                       real* __restrict__ df_tmp,
                                       real* __restrict__ dfT,
                                       real* __restrict__ dfT_tmp,
                                       const voxel_t* __restrict__ voxels,
                                       BoundaryCondition* __restrict__ bcs,
                                       const real nu,
                                       const real C,
                                       const real nuT,
                                       const real Pr_t,
                                       const real gBetta,
                                       const real Tref,
                                       real* __restrict__ averageSrc,
                                       real* __restrict__ averageDst) {
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y * (partSize.z() - 1);

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  compute(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp, dfT,
          dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref, averageSrc,
          averageDst);
}

__global__ void ComputeAndPlotKernelInterior(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
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
  Eigen::Vector3i partSize = partition.getExtents();
  Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x + partGhostLayer.x();
  const int y = blockIdx.x + partGhostLayer.y();
  const int z = blockIdx.y + partGhostLayer.z();

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}

__global__ void ComputeAndPlotKernelBoundaryX(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
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
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = blockIdx.y * (partSize.x() - 1);  // Might not be multiple of 32
  const int y = threadIdx.x;
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}

__global__ void ComputeAndPlotKernelBoundaryY(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
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
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.y * (partSize.y() - 1);
  const int z = blockIdx.x;

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}

__global__ void ComputeAndPlotKernelBoundaryZ(
    const Partition partition,
    real* __restrict__ df,
    real* __restrict__ df_tmp,
    real* __restrict__ dfT,
    real* __restrict__ dfT_tmp,
    const voxel_t* __restrict__ voxels,
    BoundaryCondition* __restrict__ bcs,
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
  const Eigen::Vector3i partSize = partition.getExtents();
  const Eigen::Vector3i partGhostLayer = partition.getGhostLayer();

  // Compute node position from thread indexes
  const int x = threadIdx.x;
  const int y = blockIdx.x;
  const int z = blockIdx.y * (partSize.z() - 1);

  // Check that the thread is inside the simulation domain
  if ((x >= partSize.x()) || (y >= partSize.y()) || (z >= partSize.z())) return;

  computeAndPlot(Eigen::Vector3i(x, y, z), partSize, partGhostLayer, df, df_tmp,
                 dfT, dfT_tmp, voxels, bcs, nu, C, nuT, Pr_t, gBetta, Tref,
                 averageSrc, averageDst, displayQuantity, plot);
}
