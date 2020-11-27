#include "InitKernel.hpp"
__global__ void InitKernel(real_t* __restrict__ df,
                           real_t* __restrict__ dfT,
                           int nx,
                           int ny,
                           int nz,
                           real_t rho,
                           real_t vx,
                           real_t vy,
                           real_t vz,
                           real_t T) {
  real_t sq_term;
  Vector3<int> pos(threadIdx.x, blockIdx.x, blockIdx.y);
  if ((pos.x() >= nx) || (pos.y() >= ny) || (pos.z() >= nz)) return;
  const int x = pos.x();
  const int y = pos.y();
  const int z = pos.z();
  sq_term = -1.5f * powf(vx, 2) - 1.5f * powf(vy, 2) - 1.5f * powf(vz, 2);
  df3D(0, x, y, z, nx, ny, nz) = 0.33333333333333331f * rho * (sq_term + 1.0f);
  df3D(1, x, y, z, nx, ny, nz) =
      0.055555555555555552f * rho *
      (sq_term + 4.5f * powf(vx, 2) + 3.0f * vx + 1.0f);
  df3D(2, x, y, z, nx, ny, nz) =
      0.055555555555555552f * rho *
      (sq_term + 4.5f * powf(vx, 2) - 3.0f * vx + 1.0f);
  df3D(3, x, y, z, nx, ny, nz) =
      0.055555555555555552f * rho *
      (sq_term + 4.5f * powf(vy, 2) + 3.0f * vy + 1.0f);
  df3D(4, x, y, z, nx, ny, nz) =
      0.055555555555555552f * rho *
      (sq_term + 4.5f * powf(vy, 2) - 3.0f * vy + 1.0f);
  df3D(5, x, y, z, nx, ny, nz) =
      0.055555555555555552f * rho *
      (sq_term + 4.5f * powf(vz, 2) + 3.0f * vz + 1.0f);
  df3D(6, x, y, z, nx, ny, nz) =
      0.055555555555555552f * rho *
      (sq_term + 4.5f * powf(vz, 2) - 3.0f * vz + 1.0f);
  df3D(7, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx + 3.0f * vy + 4.5f * powf(vx + vy, 2) + 1.0f);
  df3D(8, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx - 3.0f * vy + 4.5f * powf(-vx - vy, 2) + 1.0f);
  df3D(9, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx - 3.0f * vy + 4.5f * powf(vx - vy, 2) + 1.0f);
  df3D(10, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx + 3.0f * vy + 4.5f * powf(-vx + vy, 2) + 1.0f);
  df3D(11, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx + 3.0f * vz + 4.5f * powf(vx + vz, 2) + 1.0f);
  df3D(12, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx - 3.0f * vz + 4.5f * powf(-vx - vz, 2) + 1.0f);
  df3D(13, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vx - 3.0f * vz + 4.5f * powf(vx - vz, 2) + 1.0f);
  df3D(14, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vx + 3.0f * vz + 4.5f * powf(-vx + vz, 2) + 1.0f);
  df3D(15, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vy + 3.0f * vz + 4.5f * powf(vy + vz, 2) + 1.0f);
  df3D(16, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vy - 3.0f * vz + 4.5f * powf(-vy - vz, 2) + 1.0f);
  df3D(17, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term + 3.0f * vy - 3.0f * vz + 4.5f * powf(vy - vz, 2) + 1.0f);
  df3D(18, x, y, z, nx, ny, nz) =
      0.027777777777777776f * rho *
      (sq_term - 3.0f * vy + 3.0f * vz + 4.5f * powf(-vy + vz, 2) + 1.0f);
  Tdf3D(0, x, y, z, nx, ny, nz) = 0.14285714285714285f * T;
  Tdf3D(1, x, y, z, nx, ny, nz) = 0.14285714285714285f * T * (3.5f * vx + 1.0f);
  Tdf3D(2, x, y, z, nx, ny, nz) = 0.14285714285714285f * T * (1.0f - 3.5f * vx);
  Tdf3D(3, x, y, z, nx, ny, nz) = 0.14285714285714285f * T * (3.5f * vy + 1.0f);
  Tdf3D(4, x, y, z, nx, ny, nz) = 0.14285714285714285f * T * (1.0f - 3.5f * vy);
  Tdf3D(5, x, y, z, nx, ny, nz) = 0.14285714285714285f * T * (3.5f * vz + 1.0f);
  Tdf3D(6, x, y, z, nx, ny, nz) = 0.14285714285714285f * T * (1.0f - 3.5f * vz);
}
