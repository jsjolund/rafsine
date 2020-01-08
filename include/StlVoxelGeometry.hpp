#pragma once

#include <omp.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

#include "StlMesh.hpp"
#include "VoxelGeometry.hpp"

class StlVoxelGeometry : public VoxelGeometry {
 private:
  std::vector<stl_mesh::StlMesh*> m_meshes;

 public:
  /**
   * @brief Get the minimum and maximum vertex positions
   * @param minOut Minimum
   * @param maxOut Maximum
   */
  void getExtents(Eigen::Vector3f* minOut, Eigen::Vector3f* maxOut) {
    for (int i = 0; i < 3; ++i) {
      (*minOut)(i) = std::numeric_limits<float>::max();
      (*maxOut)(i) = std::numeric_limits<float>::min();
    }
    for (stl_mesh::StlMesh* mesh : m_meshes) {
      for (Eigen::Vector3f& v : mesh->vertices) {
        for (int i = 0; i < 3; ++i) {
          if (v(i) < (*minOut)(i)) (*minOut)(i) = v(i);
          if (v(i) > (*maxOut)(i)) (*maxOut)(i) = v(i);
        }
      }
    }
  }

  /**
   * @brief Right-multiply vertices and normals with a matrix
   * @param mat
   */
  void transform(Eigen::Matrix3f mat) {
    for (stl_mesh::StlMesh* mesh : m_meshes) {
      for (Eigen::Vector3f& v : mesh->normals) v = mat * v;
      for (Eigen::Vector3f& v : mesh->vertices) v = mat * v;
    }
  }

  /**
   * @brief Translate the vertices
   *
   * @param tra
   */
  void translate(Eigen::Vector3f tra) {
    for (stl_mesh::StlMesh* mesh : m_meshes) {
      for (Eigen::Vector3f& v : mesh->vertices) v += tra;
    }
  }

  /**
   * @brief Scales the vertices with these values
   * @param scl
   */
  void scale(Eigen::Vector3f scl) {
    for (stl_mesh::StlMesh* mesh : m_meshes) {
      for (Eigen::Vector3f& v : mesh->vertices) {
        v.x() *= scl.x();
        v.y() *= scl.y();
        v.z() *= scl.z();
      }
    }
  }

  /**
   * @brief Set the minimum and maximum vertices to
   * [0, size.x]x[0, size.y]x[0, size.z]
   * @param size
   */
  void setSize(Eigen::Vector3f size) {
    Eigen::Vector3f min, max;
    getExtents(&min, &max);
    translate(-min);
    min = Eigen::Vector3f(0, 0, 0);
    max = max - min;
    Eigen::Vector3f ratio(size.x() / max.x(), size.y() / max.y(),
                          size.z() / max.z());
    scale(ratio);
  }

  /**
   * @brief Compute triangle box intersection
   *
   * @param min Min defining voxel
   * @param max Max defining voxel
   * @param v1 First vertex
   * @param v2 Second vertex
   * @param v3 Third vertex
   * @return true The triangle intersects the box
   * @return false Otherwise
   */
  bool triangleBoxIntersection(const Eigen::Vector3f& min,
                               const Eigen::Vector3f& max,
                               const Eigen::Vector3f& v1,
                               const Eigen::Vector3f& v2,
                               const Eigen::Vector3f& v3) {
    float half_size[3] = {static_cast<float>((max(0) - min(0)) / 2.),
                          static_cast<float>((max(1) - min(1)) / 2.),
                          static_cast<float>((max(2) - min(2)) / 2.)};
    float center[3] = {max(0) - half_size[0], max(1) - half_size[1],
                       max(2) - half_size[2]};
    float vertices[3][3] = {
        {v1(0), v1(1), v1(2)}, {v2(0), v2(1), v2(2)}, {v3(0), v3(1), v3(2)}};
    return triBoxOverlap(center, half_size, vertices);
  }

  Eigen::Vector3i discretize(Eigen::Vector3f v) {
    Eigen::Vector3i u;
    for (int i = 0; i < 3; i++)
      u[i] = static_cast<int>(v[i] > 0 ? v[i] + 0.5 : v[i] - 0.5);
    return u;
  }

  void voxelize() {
    // TODO(This does not work yet)
    const int height = getSizeX();
    const int width = getSizeY();
    const int depth = getSizeZ();

    for (int j = 0; j < m_meshes.size(); j++) {
      stl_mesh::StlMesh* mesh = m_meshes.at(j);
      std::string name = mesh->name;

      for (int k = 0; k < mesh->vertices.size(); k += 3) {
        Eigen::Vector3f n = mesh->normals.at(k / 3);
        Eigen::Vector3f v1 = mesh->vertices.at(k + 0);
        Eigen::Vector3f v2 = mesh->vertices.at(k + 1);
        Eigen::Vector3f v3 = mesh->vertices.at(k + 2);

#pragma omp parallel
        {
#pragma omp for
          for (int i = 0; i < height * width * depth; i++) {
            const int d = i % depth;
            const int w = (i / depth) % width;
            const int h = (i / depth) / width;

            Eigen::Vector3f min(w, h, d);
            Eigen::Vector3f max(w + 1, h + 1, d + 1);

            bool overlap = triangleBoxIntersection(min, max, v1, v2, v3);
            if (overlap) {
#pragma omp critical
              {
                BoundaryCondition bc;
                bc.m_type = VoxelType::Enum::WALL;
                bc.m_normal = discretize(n);
                Eigen::Vector3i p(h + 1, w + 1, d + 1);
                storeType(&bc, name);
                set(p, bc, NodeMode::Enum::INTERSECT, name);
              }
            }
          }
        }
      }
    }
  }

  StlVoxelGeometry(
      int nx,
      int ny,
      int nz,
      std::vector<stl_mesh::StlMesh*> meshes,
      const Eigen::Matrix3f globalTransform = Eigen::Matrix3f::Identity())
      : VoxelGeometry(nx, ny, nz), m_meshes(meshes) {
    transform(globalTransform);
    setSize(Eigen::Vector3f(nx, ny, nz));
  }
};
