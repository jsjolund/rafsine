#pragma once

#include <limits>
#include <vector>

#include "StlMesh.hpp"
#include "VoxelGeometry.hpp"

class StlVoxelMesh : public VoxelGeometry {
 private:
  const std::vector<stl_mesh::StlMesh> m_meshes;

 public:
  void getExtents(Eigen::Vector3f *minOut, Eigen::Vector3f *maxOut) {
    minOut->x() = std::numeric_limits<float>::max();
    minOut->y() = std::numeric_limits<float>::max();
    minOut->z() = std::numeric_limits<float>::max();
    maxOut->x() = std::numeric_limits<float>::min();
    maxOut->y() = std::numeric_limits<float>::min();
    maxOut->z() = std::numeric_limits<float>::min();

    for (stl_mesh::StlMesh mesh : m_meshes) {
      for (int i = 0; i < mesh.vertices.size(); i += 3) {
        float x = mesh.vertices.at(i + 0);
        float y = mesh.vertices.at(i + 1);
        float z = mesh.vertices.at(i + 2);

        if (x < minOut->x()) minOut->x() = x;
        if (y < minOut->y()) minOut->y() = y;
        if (z < minOut->z()) minOut->z() = z;
        if (x > maxOut->x()) maxOut->x() = x;
        if (y > maxOut->y()) maxOut->y() = y;
        if (z > maxOut->z()) maxOut->z() = z;
      }
    }
  }

  StlVoxelMesh(int nx, int ny, int nz,
               const std::vector<stl_mesh::StlMesh> &meshes)
      : VoxelGeometry(nx, ny, nz), m_meshes(meshes) {}
};
