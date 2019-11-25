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

  real rho;
  real T;
  real vx, vy, vz;
  real f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq, f9eq, f10eq, f11eq,
      f12eq, f13eq, f14eq, f15eq, f16eq, f17eq, f18eq;
  real m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15,
      m16, m17, m18;
  real m0eq, m1eq, m2eq, m3eq, m4eq, m5eq, m6eq, m7eq, m8eq, m9eq, m10eq, m11eq,
      m12eq, m13eq, m14eq, m15eq, m16eq, m17eq, m18eq;
  real m0neq, m1neq, m2neq, m3neq, m4neq, m5neq, m6neq, m7neq, m8neq, m9neq,
      m10neq, m11neq, m12neq, m13neq, m14neq, m15neq, m16neq, m17neq, m18neq;
  real omega0, omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8,
      omega9, omega10, omega11, omega12, omega13, omega14, omega15, omega16,
      omega17, omega18;
  rho = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
        f3 + f4 + f5 + f6 + f7 + f8 + f9;
  T = T0 + T1 + T2 + T3 + T4 + T5 + T6;
  vx = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
  vy = f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9;
  vz = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6;
  f0eq = 0.33333333333333331 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f1eq = 0.055555555555555552 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f2eq = 0.055555555555555552 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f3eq = 0.055555555555555552 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f4eq = 0.055555555555555552 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f5eq = 0.055555555555555552 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f6eq = 0.055555555555555552 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f7eq = 0.027777777777777776 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f8eq = 0.027777777777777776 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f9eq = 0.027777777777777776 * rho *
         (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f10eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f11eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f12eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f13eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f14eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f15eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f16eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f17eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  f18eq = 0.027777777777777776 * rho *
          (-1.5 * pow(vx, 2) - 1.5 * pow(vy, 2) - 1.5 * pow(vz, 2) + 1);
  m0 = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 + f3 +
       f4 + f5 + f6 + f7 + f8 + f9;
  m1 = -30 * f0 - 11 * f1 + 8 * f10 + 8 * f11 + 8 * f12 + 8 * f13 + 8 * f14 +
       8 * f15 + 8 * f16 + 8 * f17 + 8 * f18 - 11 * f2 - 11 * f3 - 11 * f4 -
       11 * f5 - 11 * f6 + 8 * f7 + 8 * f8 + 8 * f9;
  m2 = 12 * f0 - 4 * f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 -
       4 * f2 - 4 * f3 - 4 * f4 - 4 * f5 - 4 * f6 + f7 + f8 + f9;
  m3 = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
  m4 = -4 * f1 - f10 + f11 - f12 + f13 - f14 + 4 * f2 + f7 - f8 + f9;
  m5 = f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9;
  m6 = f10 + f15 - f16 + f17 - f18 - 4 * f3 + 4 * f4 + f7 - f8 - f9;
  m7 = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6;
  m8 = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 - 4 * f5 + 4 * f6;
  m9 = 2 * f1 + f10 + f11 + f12 + f13 + f14 - 2 * f15 - 2 * f16 - 2 * f17 -
       2 * f18 + 2 * f2 - f3 - f4 - f5 - f6 + f7 + f8 + f9;
  m10 = -4 * f1 + f10 + f11 + f12 + f13 + f14 - 2 * f15 - 2 * f16 - 2 * f17 -
        2 * f18 - 4 * f2 + 2 * f3 + 2 * f4 + 2 * f5 + 2 * f6 + f7 + f8 + f9;
  m11 = f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9;
  m12 = f10 - f11 - f12 - f13 - f14 - 2 * f3 - 2 * f4 + 2 * f5 + 2 * f6 + f7 +
        f8 + f9;
  m13 = -f10 + f7 + f8 - f9;
  m14 = f15 + f16 - f17 - f18;
  m15 = f11 + f12 - f13 - f14;
  m16 = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9;
  m17 = -f10 + f15 - f16 + f17 - f18 - f7 + f8 + f9;
  m18 = f11 - f12 - f13 + f14 - f15 + f16 + f17 - f18;
  m0eq = f0 + f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f2 +
         f3 + f4 + f5 + f6 + f7 + f8 + f9;
  m1eq = -11 * f0 - 11 * f1 - 11 * f10 - 11 * f11 - 11 * f12 - 11 * f13 -
         11 * f14 - 11 * f15 - 11 * f16 - 11 * f17 - 11 * f18 - 11 * f2 -
         11 * f3 - 11 * f4 - 11 * f5 - 11 * f6 - 11 * f7 - 11 * f8 - 11 * f9 +
         19.0 * pow(f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9, 2) +
         19.0 * pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) +
         19.0 * pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m2eq = -7.5396825396825395 *
             pow(f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9, 2) -
         7.5396825396825395 *
             pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) -
         7.5396825396825395 *
             pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m3eq = f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9;
  m4eq = -0.66666666666666663 * f1 + 0.66666666666666663 * f10 -
         0.66666666666666663 * f11 + 0.66666666666666663 * f12 -
         0.66666666666666663 * f13 + 0.66666666666666663 * f14 +
         0.66666666666666663 * f2 - 0.66666666666666663 * f7 +
         0.66666666666666663 * f8 - 0.66666666666666663 * f9;
  m5eq = f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9;
  m6eq = -0.66666666666666663 * f10 - 0.66666666666666663 * f15 +
         0.66666666666666663 * f16 - 0.66666666666666663 * f17 +
         0.66666666666666663 * f18 - 0.66666666666666663 * f3 +
         0.66666666666666663 * f4 - 0.66666666666666663 * f7 +
         0.66666666666666663 * f8 + 0.66666666666666663 * f9;
  m7eq = f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6;
  m8eq = -0.66666666666666663 * f11 + 0.66666666666666663 * f12 +
         0.66666666666666663 * f13 - 0.66666666666666663 * f14 -
         0.66666666666666663 * f15 + 0.66666666666666663 * f16 +
         0.66666666666666663 * f17 - 0.66666666666666663 * f18 -
         0.66666666666666663 * f5 + 0.66666666666666663 * f6;
  m9eq = 1.0 * (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) *
             (2 * f1 - 2 * f10 + 2 * f11 - 2 * f12 + 2 * f13 - 2 * f14 -
              2 * f2 + 2 * f7 - 2 * f8 + 2 * f9) -
         1.0 * pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) -
         1.0 * pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m10eq = 0;
  m11eq = 1.0 * pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) -
          1.0 * pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m12eq = 0;
  m13eq = (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) *
          (f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9);
  m14eq = (f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9) *
          (f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6);
  m15eq = (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) *
          (f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6);
  m16eq = 0;
  m17eq = 0;
  m18eq = 0;
  m0neq = 0;
  m1neq =
      -19 * f0 + 19 * f10 + 19 * f11 + 19 * f12 + 19 * f13 + 19 * f14 +
      19 * f15 + 19 * f16 + 19 * f17 + 19 * f18 + 19 * f7 + 19 * f8 + 19 * f9 -
      19.0 * pow(f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9, 2) -
      19.0 * pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) -
      19.0 * pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m2neq = 12 * f0 - 4 * f1 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 +
          f18 - 4 * f2 - 4 * f3 - 4 * f4 - 4 * f5 - 4 * f6 + f7 + f8 + f9 +
          7.5396825396825395 *
              pow(f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9, 2) +
          7.5396825396825395 *
              pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) +
          7.5396825396825395 *
              pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m3neq = 0;
  m4neq = -3.3333333333333335 * f1 - 1.6666666666666665 * f10 +
          1.6666666666666665 * f11 - 1.6666666666666665 * f12 +
          1.6666666666666665 * f13 - 1.6666666666666665 * f14 +
          3.3333333333333335 * f2 + 1.6666666666666665 * f7 -
          1.6666666666666665 * f8 + 1.6666666666666665 * f9;
  m5neq = 0;
  m6neq = 1.6666666666666665 * f10 + 1.6666666666666665 * f15 -
          1.6666666666666665 * f16 + 1.6666666666666665 * f17 -
          1.6666666666666665 * f18 - 3.3333333333333335 * f3 +
          3.3333333333333335 * f4 + 1.6666666666666665 * f7 -
          1.6666666666666665 * f8 - 1.6666666666666665 * f9;
  m7neq = 0;
  m8neq = 1.6666666666666665 * f11 - 1.6666666666666665 * f12 -
          1.6666666666666665 * f13 + 1.6666666666666665 * f14 +
          1.6666666666666665 * f15 - 1.6666666666666665 * f16 -
          1.6666666666666665 * f17 + 1.6666666666666665 * f18 -
          3.3333333333333335 * f5 + 3.3333333333333335 * f6;
  m9neq = 2 * f1 + f10 + f11 + f12 + f13 + f14 - 2 * f15 - 2 * f16 - 2 * f17 -
          2 * f18 + 2 * f2 - f3 - f4 - f5 - f6 + f7 + f8 + f9 -
          1.0 * (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) *
              (2 * f1 - 2 * f10 + 2 * f11 - 2 * f12 + 2 * f13 - 2 * f14 -
               2 * f2 + 2 * f7 - 2 * f8 + 2 * f9) +
          1.0 * pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) +
          1.0 * pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m10neq = -4 * f1 + f10 + f11 + f12 + f13 + f14 - 2 * f15 - 2 * f16 - 2 * f17 -
           2 * f18 - 4 * f2 + 2 * f3 + 2 * f4 + 2 * f5 + 2 * f6 + f7 + f8 + f9;
  m11neq =
      f10 - f11 - f12 - f13 - f14 + f3 + f4 - f5 - f6 + f7 + f8 + f9 -
      1.0 * pow(f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9, 2) +
      1.0 * pow(f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6, 2);
  m12neq = f10 - f11 - f12 - f13 - f14 - 2 * f3 - 2 * f4 + 2 * f5 + 2 * f6 +
           f7 + f8 + f9;
  m13neq = -f10 + f7 + f8 - f9 -
           (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) *
               (f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9);
  m14neq = f15 + f16 - f17 - f18 -
           (f10 + f15 - f16 + f17 - f18 + f3 - f4 + f7 - f8 - f9) *
               (f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6);
  m15neq = f11 + f12 - f13 - f14 -
           (f1 - f10 + f11 - f12 + f13 - f14 - f2 + f7 - f8 + f9) *
               (f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18 + f5 - f6);
  m16neq = -f10 - f11 + f12 - f13 + f14 + f7 - f8 + f9;
  m17neq = -f10 + f15 - f16 + f17 - f18 - f7 + f8 + f9;
  m18neq = f11 - f12 - f13 + f14 - f15 + f16 + f17 - f18;
  omega0 = 0.014912280701754384 * m1neq - 0.066666666666666652 * m2neq;
  omega1 = 0.077777777777777765 * m10neq + 0.0054678362573099409 * m1neq +
           0.02222222222222222 * m2neq + 0.12 * m4neq -
           1.0 / 9.0 * m9neq / (6 * nu + 1);
  omega2 = 0.077777777777777765 * m10neq + 0.0054678362573099409 * m1neq +
           0.02222222222222222 * m2neq - 0.12 * m4neq -
           1.0 / 9.0 * m9neq / (6 * nu + 1);
  omega3 = -0.038888888888888883 * m10neq - 1.0 / 6.0 * m11neq / (6 * nu + 1) +
           0.11666666666666665 * m12neq + 0.0054678362573099409 * m1neq +
           0.02222222222222222 * m2neq + 0.12 * m6neq +
           (1.0 / 18.0) * m9neq / (6 * nu + 1);
  omega4 = -0.038888888888888883 * m10neq - 1.0 / 6.0 * m11neq / (6 * nu + 1) +
           0.11666666666666665 * m12neq + 0.0054678362573099409 * m1neq +
           0.02222222222222222 * m2neq - 0.12 * m6neq +
           (1.0 / 18.0) * m9neq / (6 * nu + 1);
  omega5 = -0.038888888888888883 * m10neq +
           (1.0 / 6.0) * m11neq / (6 * nu + 1) - 0.11666666666666665 * m12neq +
           0.0054678362573099409 * m1neq + 0.02222222222222222 * m2neq +
           0.12 * m8neq + (1.0 / 18.0) * m9neq / (6 * nu + 1);
  omega6 = -0.038888888888888883 * m10neq +
           (1.0 / 6.0) * m11neq / (6 * nu + 1) - 0.11666666666666665 * m12neq +
           0.0054678362573099409 * m1neq + 0.02222222222222222 * m2neq -
           0.12 * m8neq + (1.0 / 18.0) * m9neq / (6 * nu + 1);
  omega7 = -0.019444444444444441 * m10neq - 1.0 / 6.0 * m11neq / (6 * nu + 1) -
           0.058333333333333327 * m12neq - 1.0 / 2.0 * m13neq / (6 * nu + 1) -
           0.2475 * m16neq + 0.2475 * m17neq - 0.0039766081871345027 * m1neq -
           0.0055555555555555549 * m2neq - 0.029999999999999999 * m4neq -
           0.029999999999999999 * m6neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega8 = -0.019444444444444441 * m10neq - 1.0 / 6.0 * m11neq / (6 * nu + 1) -
           0.058333333333333327 * m12neq - 1.0 / 2.0 * m13neq / (6 * nu + 1) +
           0.2475 * m16neq - 0.2475 * m17neq - 0.0039766081871345027 * m1neq -
           0.0055555555555555549 * m2neq + 0.029999999999999999 * m4neq +
           0.029999999999999999 * m6neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega9 = -0.019444444444444441 * m10neq - 1.0 / 6.0 * m11neq / (6 * nu + 1) -
           0.058333333333333327 * m12neq + (1.0 / 2.0) * m13neq / (6 * nu + 1) -
           0.2475 * m16neq - 0.2475 * m17neq - 0.0039766081871345027 * m1neq -
           0.0055555555555555549 * m2neq - 0.029999999999999999 * m4neq +
           0.029999999999999999 * m6neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega10 = -0.019444444444444441 * m10neq - 1.0 / 6.0 * m11neq / (6 * nu + 1) -
            0.058333333333333327 * m12neq +
            (1.0 / 2.0) * m13neq / (6 * nu + 1) + 0.2475 * m16neq +
            0.2475 * m17neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq + 0.029999999999999999 * m4neq -
            0.029999999999999999 * m6neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega11 = -0.019444444444444441 * m10neq +
            (1.0 / 6.0) * m11neq / (6 * nu + 1) +
            0.058333333333333327 * m12neq - 1.0 / 2.0 * m15neq / (6 * nu + 1) +
            0.2475 * m16neq - 0.2475 * m18neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq - 0.029999999999999999 * m4neq -
            0.029999999999999999 * m8neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega12 = -0.019444444444444441 * m10neq +
            (1.0 / 6.0) * m11neq / (6 * nu + 1) +
            0.058333333333333327 * m12neq - 1.0 / 2.0 * m15neq / (6 * nu + 1) -
            0.2475 * m16neq + 0.2475 * m18neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq + 0.029999999999999999 * m4neq +
            0.029999999999999999 * m8neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega13 =
      -0.019444444444444441 * m10neq + (1.0 / 6.0) * m11neq / (6 * nu + 1) +
      0.058333333333333327 * m12neq + (1.0 / 2.0) * m15neq / (6 * nu + 1) +
      0.2475 * m16neq + 0.2475 * m18neq - 0.0039766081871345027 * m1neq -
      0.0055555555555555549 * m2neq - 0.029999999999999999 * m4neq +
      0.029999999999999999 * m8neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega14 =
      -0.019444444444444441 * m10neq + (1.0 / 6.0) * m11neq / (6 * nu + 1) +
      0.058333333333333327 * m12neq + (1.0 / 2.0) * m15neq / (6 * nu + 1) -
      0.2475 * m16neq - 0.2475 * m18neq - 0.0039766081871345027 * m1neq -
      0.0055555555555555549 * m2neq + 0.029999999999999999 * m4neq -
      0.029999999999999999 * m8neq - 1.0 / 18.0 * m9neq / (6 * nu + 1);
  omega15 = 0.038888888888888883 * m10neq - 1.0 / 2.0 * m14neq / (6 * nu + 1) -
            0.2475 * m17neq + 0.2475 * m18neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq - 0.029999999999999999 * m6neq -
            0.029999999999999999 * m8neq + (1.0 / 9.0) * m9neq / (6 * nu + 1);
  omega16 = 0.038888888888888883 * m10neq - 1.0 / 2.0 * m14neq / (6 * nu + 1) +
            0.2475 * m17neq - 0.2475 * m18neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq + 0.029999999999999999 * m6neq +
            0.029999999999999999 * m8neq + (1.0 / 9.0) * m9neq / (6 * nu + 1);
  omega17 = 0.038888888888888883 * m10neq +
            (1.0 / 2.0) * m14neq / (6 * nu + 1) - 0.2475 * m17neq -
            0.2475 * m18neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq - 0.029999999999999999 * m6neq +
            0.029999999999999999 * m8neq + (1.0 / 9.0) * m9neq / (6 * nu + 1);
  omega18 = 0.038888888888888883 * m10neq +
            (1.0 / 2.0) * m14neq / (6 * nu + 1) + 0.2475 * m17neq +
            0.2475 * m18neq - 0.0039766081871345027 * m1neq -
            0.0055555555555555549 * m2neq + 0.029999999999999999 * m6neq -
            0.029999999999999999 * m8neq + (1.0 / 9.0) * m9neq / (6 * nu + 1);
  dftmp3D(0, x, y, z, nx, ny, nz) = f0 + omega0;
  dftmp3D(1, x, y, z, nx, ny, nz) = f1 + omega1;
  dftmp3D(2, x, y, z, nx, ny, nz) = f2 + omega2;
  dftmp3D(3, x, y, z, nx, ny, nz) = f3 + omega3;
  dftmp3D(4, x, y, z, nx, ny, nz) = f4 + omega4;
  dftmp3D(5, x, y, z, nx, ny, nz) = f5 + omega5;
  dftmp3D(6, x, y, z, nx, ny, nz) = f6 + omega6;
  dftmp3D(7, x, y, z, nx, ny, nz) = f7 + omega7;
  dftmp3D(8, x, y, z, nx, ny, nz) = f8 + omega8;
  dftmp3D(9, x, y, z, nx, ny, nz) = f9 + omega9;
  dftmp3D(10, x, y, z, nx, ny, nz) = f10 + omega10;
  dftmp3D(11, x, y, z, nx, ny, nz) = f11 + omega11;
  dftmp3D(12, x, y, z, nx, ny, nz) = f12 + omega12;
  dftmp3D(13, x, y, z, nx, ny, nz) = f13 + omega13;
  dftmp3D(14, x, y, z, nx, ny, nz) = f14 + omega14;
  dftmp3D(15, x, y, z, nx, ny, nz) = f15 + omega15;
  dftmp3D(16, x, y, z, nx, ny, nz) = f16 + omega16;
  dftmp3D(17, x, y, z, nx, ny, nz) = f17 + omega17;
  dftmp3D(18, x, y, z, nx, ny, nz) = f18 + omega18;
  Tdftmp3D(0, x, y, z, nx, ny, nz) = T0;
  Tdftmp3D(1, x, y, z, nx, ny, nz) = T1;
  Tdftmp3D(2, x, y, z, nx, ny, nz) = T2;
  Tdftmp3D(3, x, y, z, nx, ny, nz) = T3;
  Tdftmp3D(4, x, y, z, nx, ny, nz) = T4;
  Tdftmp3D(5, x, y, z, nx, ny, nz) = T5;
  Tdftmp3D(6, x, y, z, nx, ny, nz) = T6;

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
