#include "VoxelObject.hpp"

unsigned int VoxelSphere::idx(int x, int y, int z) {
  return x + y * m_n + z * m_n * m_n;
}

unsigned int VoxelSphere::idxn(int x, int y, int z) {
  return idx(x + m_n / 2, y + m_n / 2, z + m_n / 2);
}

void VoxelSphere::fill(const int x, const int y, const int z) {
  m_grid.at(idxn(x, y, z)) = SphereVoxel::Enum::SURFACE;
}

void VoxelSphere::fillInside(const int x, const int y, const int z) {
  int ax = abs(x);
  int ay = abs(y);
  int az = abs(z);
  int bx = -ax;
  int by = -ay;
  int bz = -az;
  for (int ix = ax; ix >= bx; ix--) {
    for (int iy = ay; iy >= by; iy--) {
      for (int iz = az; iz >= bz; iz--) {
        if (m_grid.at(idxn(ix, iy, iz)) != SphereVoxel::Enum::SURFACE)
          m_grid.at(idxn(ix, iy, iz)) = SphereVoxel::Enum::INSIDE;
      }
    }
  }
}

void VoxelSphere::fillSigns(int x, int y, int z) {
  fill(x, y, z);
  for (;;) {
    if ((z = -z) >= 0) {
      if ((y = -y) >= 0) {
        if ((x = -x) >= 0) { break; }
      }
    }
    fill(x, y, z);
  }
  fillInside(x, y, z);
}

void VoxelSphere::fillAll(int x, int y, int z) {
  fillSigns(x, y, z);
  if (z > y) { fillSigns(x, z, y); }
  if (z > x && z > y) { fillSigns(z, y, x); }
}

SphereVoxel::Enum VoxelSphere::getVoxel(unsigned int x,
                                        unsigned int y,
                                        unsigned int z) {
  try {
    return m_grid.at(idx(x, y, z));
  } catch (const std::exception e) { return SphereVoxel::Enum::OUTSIDE; }
}

vector3<int> VoxelSphere::getNormal(unsigned int x,
                                    unsigned int y,
                                    unsigned int z) {
  return m_normals.at(idx(x, y, z));
}

void VoxelSphere::createSphere(float R) {
  std::fill(m_grid.begin(), m_grid.end(), SphereVoxel::Enum::OUTSIDE);
  std::fill(m_normals.begin(), m_normals.end(), vector3<int>(0, 0, 0));

  const int maxR2 = floor(R * R);
  int zx = floor(R);
  for (int x = 0;; ++x) {
    while (x * x + zx * zx > maxR2 && zx >= x) --zx;
    if (zx < x) break;
    int z = zx;
    for (int y = 0;; ++y) {
      while (x * x + y * y + z * z > maxR2 && z >= x && z >= y) --z;
      if (z < x || z < y) break;
      fillAll(x, y, z);
    }
  }
  std::vector<SphereVoxel::Enum> cornerGrid(m_n * m_n * m_n);
  std::fill(cornerGrid.begin(), cornerGrid.end(), SphereVoxel::Enum::OUTSIDE);
  for (unsigned int x = 0; x < m_n; x++)
    for (unsigned int y = 0; y < m_n; y++)
      for (unsigned int z = 0; z < m_n; z++) {
        if (getVoxel(x, y, z) == SphereVoxel::Enum::INSIDE) {
          int adjacent = 0;
          if (getVoxel(x + 1, y, z) == SphereVoxel::Enum::SURFACE) adjacent++;
          if (getVoxel(x - 1, y, z) == SphereVoxel::Enum::SURFACE) adjacent++;
          if (getVoxel(x, y + 1, z) == SphereVoxel::Enum::SURFACE) adjacent++;
          if (getVoxel(x, y - 1, z) == SphereVoxel::Enum::SURFACE) adjacent++;
          if (getVoxel(x, y, z + 1) == SphereVoxel::Enum::SURFACE) adjacent++;
          if (getVoxel(x, y, z - 1) == SphereVoxel::Enum::SURFACE) adjacent++;
          if (adjacent > 1)
            cornerGrid.at(idx(x, y, z)) = SphereVoxel::Enum::CORNER;
        }
      }

  for (unsigned int x = 0; x < m_n; x++)
    for (unsigned int y = 0; y < m_n; y++)
      for (unsigned int z = 0; z < m_n; z++) {
        if (cornerGrid.at(idx(x, y, z)) == SphereVoxel::Enum::CORNER)
          m_grid.at(idx(x, y, z)) = SphereVoxel::Enum::CORNER;
      }

  for (unsigned int x = 0; x < m_n; x++)
    for (unsigned int y = 0; y < m_n; y++)
      for (unsigned int z = 0; z < m_n; z++)
        if (getVoxel(x, y, z) == SphereVoxel::Enum::SURFACE) {
          if (getVoxel(x + 1, y, z) == SphereVoxel::Enum::OUTSIDE)
            m_normals.at(idx(x, y, z)) += vector3<int>(1, 0, 0);
          if (getVoxel(x - 1, y, z) == SphereVoxel::Enum::OUTSIDE)
            m_normals.at(idx(x, y, z)) += vector3<int>(-1, 0, 0);
          if (getVoxel(x, y + 1, z) == SphereVoxel::Enum::OUTSIDE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, 1, 0);
          if (getVoxel(x, y - 1, z) == SphereVoxel::Enum::OUTSIDE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, -1, 0);
          if (getVoxel(x, y, z + 1) == SphereVoxel::Enum::OUTSIDE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, 1);
          if (getVoxel(x, y, z - 1) == SphereVoxel::Enum::OUTSIDE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, -1);

        } else if (getVoxel(x, y, z) == SphereVoxel::Enum::CORNER) {
          if (getVoxel(x + 1, y, z) == SphereVoxel::Enum::SURFACE)
            m_normals.at(idx(x, y, z)) += vector3<int>(1, 0, 0);
          if (getVoxel(x - 1, y, z) == SphereVoxel::Enum::SURFACE)
            m_normals.at(idx(x, y, z)) += vector3<int>(-1, 0, 0);
          if (getVoxel(x, y + 1, z) == SphereVoxel::Enum::SURFACE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, 1, 0);
          if (getVoxel(x, y - 1, z) == SphereVoxel::Enum::SURFACE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, -1, 0);
          if (getVoxel(x, y, z + 1) == SphereVoxel::Enum::SURFACE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, 1);
          if (getVoxel(x, y, z - 1) == SphereVoxel::Enum::SURFACE)
            m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, -1);
        }
}

VoxelSphere::VoxelSphere(std::string name,
                         vector3<int> voxOrigin,
                         vector3<real_t> origin,
                         real_t radius,
                         real_t temperature)
    : VoxelObject(name),
      m_n(floor(radius) * 2 + 2),
      m_grid(m_n * m_n * m_n),
      m_normals(m_n * m_n * m_n),
      m_origin(origin),
      m_voxOrigin(voxOrigin),
      m_radius(radius),
      m_voxRadius(floor(radius) + 1),
      m_temperature(temperature) {
  createSphere(radius);
}
