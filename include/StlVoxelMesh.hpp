#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "StlMesh.hpp"
#include "VoxelGeometry.hpp"

class StlVoxelMesh : public VoxelGeometry {
 private:
  std::vector<stl_mesh::StlMesh *> m_meshes;

 public:
  void getExtents(Eigen::Vector3f *minOut, Eigen::Vector3f *maxOut) {
    minOut->x() = std::numeric_limits<float>::max();
    minOut->y() = std::numeric_limits<float>::max();
    minOut->z() = std::numeric_limits<float>::max();
    maxOut->x() = std::numeric_limits<float>::min();
    maxOut->y() = std::numeric_limits<float>::min();
    maxOut->z() = std::numeric_limits<float>::min();

    for (stl_mesh::StlMesh *mesh : m_meshes) {
      for (Eigen::Vector3f &v : mesh->vertices) {
        if (v.x() < minOut->x()) minOut->x() = v.x();
        if (v.y() < minOut->y()) minOut->y() = v.y();
        if (v.z() < minOut->z()) minOut->z() = v.z();
        if (v.x() > maxOut->x()) maxOut->x() = v.x();
        if (v.y() > maxOut->y()) maxOut->y() = v.y();
        if (v.z() > maxOut->z()) maxOut->z() = v.z();
      }
    }
  }

  void transform(Eigen::Matrix3f mat) {
    for (stl_mesh::StlMesh *mesh : m_meshes) {
      for (Eigen::Vector3f &v : mesh->normals) v = mat * v;
      for (Eigen::Vector3f &v : mesh->vertices) v = mat * v;
    }
  }

  void translate(Eigen::Vector3f tra) {
    for (stl_mesh::StlMesh *mesh : m_meshes) {
      for (Eigen::Vector3f &v : mesh->vertices) v += tra;
    }
  }

  void scale(Eigen::Vector3f scl) {
    for (stl_mesh::StlMesh *mesh : m_meshes) {
      for (Eigen::Vector3f &v : mesh->vertices) {
        v.x() *= scl.x();
        v.y() *= scl.y();
        v.z() *= scl.z();
      }
    }
  }

  void setScale(Eigen::Vector3f scl) {
    Eigen::Vector3f min, max;
    getExtents(&min, &max);
    translate(-min);
    min = Eigen::Vector3f(0, 0, 0);
    max = max - min;
    Eigen::Vector3f scl(scl.x() / max.x(), scl.y() / max.y(),
                        scl.z() / max.z());
    scale(scl);
  }

  StlVoxelMesh(int nx, int ny, int nz, std::vector<stl_mesh::StlMesh *> meshes,
               Eigen::Matrix3f globalTransform = Eigen::Matrix3f::Identity())
      : VoxelGeometry(nx, ny, nz), m_meshes(meshes) {
    transform(globalTransform);
    setScale(Eigen::Vector3f(nx, ny, nz));
  }
};
