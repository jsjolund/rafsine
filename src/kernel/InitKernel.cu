#include "InitKernel.hpp"

__global__ void InitKernel(real* __restrict__ df,
                           real* __restrict__ dfT,
                           int nx,
                           int ny,
                           int nz,
                           float rho,
                           float vx,
                           float vy,
                           float vz,
                           float T,
                           float sq_term) {
  vector3<int> pos(threadIdx.x, blockIdx.x, blockIdx.y);
  if ((pos.x() >= nx) || (pos.y() >= ny) || (pos.z() >= nz)) return;
  const int x = pos.x();
  const int y = pos.y();
  const int z = pos.z();
  df3D(0, x, y, z, nx, ny, nz) = rho * (1.f / 3.f) * (1 + sq_term);
  df3D(1, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 + 3.f * vx + 4.5f * vx * vx + sq_term);
  df3D(2, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 - 3.f * vx + 4.5f * vx * vx + sq_term);
  df3D(3, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 + 3.f * vy + 4.5f * vy * vy + sq_term);
  df3D(4, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 - 3.f * vy + 4.5f * vy * vy + sq_term);
  df3D(5, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 + 3.f * vz + 4.5f * vz * vz + sq_term);
  df3D(6, x, y, z, nx, ny, nz) =
      rho * (1.f / 18.f) * (1 - 3.f * vz + 4.5f * vz * vz + sq_term);
  df3D(7, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  df3D(8, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx + vy) + 4.5f * (vx + vy) * (vx + vy) + sq_term);
  df3D(9, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  df3D(10, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx - vy) + 4.5f * (vx - vy) * (vx - vy) + sq_term);
  df3D(11, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  df3D(12, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx + vz) + 4.5f * (vx + vz) * (vx + vz) + sq_term);
  df3D(13, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  df3D(14, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vx - vz) + 4.5f * (vx - vz) * (vx - vz) + sq_term);
  df3D(15, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  df3D(16, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vy + vz) + 4.5f * (vy + vz) * (vy + vz) + sq_term);
  df3D(17, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 + 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);
  df3D(18, x, y, z, nx, ny, nz) =
      rho * (1.f / 36.f) *
      (1 - 3.f * (vy - vz) + 4.5f * (vy - vz) * (vy - vz) + sq_term);

  Tdf3D(0, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1);
  Tdf3D(1, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vx);
  Tdf3D(2, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vx);
  Tdf3D(3, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vy);
  Tdf3D(4, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vy);
  Tdf3D(5, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 + (7.f / 2.f) * vz);
  Tdf3D(6, x, y, z, nx, ny, nz) = T * (1.f / 7.f) * (1 - (7.f / 2.f) * vz);
}
