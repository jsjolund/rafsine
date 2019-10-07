#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include <omp.h>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

#include "StlMesh.hpp"
#include "VoxelGeometry.hpp"

class StlVoxelGeometry : public VoxelGeometry {
 private:
  std::vector<stl_mesh::StlMesh *> m_meshes;

 public:
  void getExtents(Eigen::Vector3f *minOut, Eigen::Vector3f *maxOut) {
    for (int i = 0; i < 3; ++i) {
      (*minOut)(i) = std::numeric_limits<float>::max();
      (*maxOut)(i) = std::numeric_limits<float>::min();
    }
    for (stl_mesh::StlMesh *mesh : m_meshes) {
      for (Eigen::Vector3f &v : mesh->vertices) {
        for (int i = 0; i < 3; ++i) {
          if (v(i) < (*minOut)(i)) (*minOut)(i) = v(i);
          if (v(i) > (*maxOut)(i)) (*maxOut)(i) = v(i);
        }
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
    Eigen::Vector3f ratio(scl.x() / max.x(), scl.y() / max.y(),
                          scl.z() / max.z());
    scale(ratio);
  }

  /** \brief Compute triangle box intersection.
   * \param[in] min defining voxel
   * \param[in] max defining voxel
   * \param[in] v1 first vertex
   * \param[in] v2 second vertex
   * \param[in] v3 third vertex
   * \return intersects
   */
  bool triangle_box_intersection(const Eigen::Vector3f &min,
                                 const Eigen::Vector3f &max,
                                 const Eigen::Vector3f &v1,
                                 const Eigen::Vector3f &v2,
                                 const Eigen::Vector3f &v3) {
    float half_size[3] = {static_cast<float>((max(0) - min(0)) / 2.),
                          static_cast<float>((max(1) - min(1)) / 2.),
                          static_cast<float>((max(2) - min(2)) / 2.)};

    float center[3] = {max(0) - half_size[0], max(1) - half_size[1],
                       max(2) - half_size[2]};

    float vertices[3][3] = {
        {v1(0), v1(1), v1(2)}, {v2(0), v2(1), v2(2)}, {v3(0), v3(1), v3(2)}};
    return triBoxOverlap(center, half_size, vertices);
  }

  void voxelize() {
    int height = getSizeX();
    int width = getSizeY();
    int depth = getSizeZ();
#pragma omp parallel
    {
#pragma omp for
      for (int i = 0; i < height * width * depth; i++) {
        int d = i % depth;
        int w = (i / depth) % width;
        int h = (i / depth) / width;

        Eigen::Vector3f min(w, h, d);
        Eigen::Vector3f max(w + 1, h + 1, d + 1);

        for (int j = 0; j < m_meshes.size(); j++) {
          stl_mesh::StlMesh *mesh = m_meshes.at(j);

          for (int k = 0; k < mesh->vertices.size(); k += 3) {
            Eigen::Vector3f v1 = mesh->vertices.at(k + 0);
            Eigen::Vector3f v2 = mesh->vertices.at(k + 1);
            Eigen::Vector3f v3 = mesh->vertices.at(k + 2);

            bool overlap = triangle_box_intersection(min, max, v1, v2, v3);
            if (overlap) {
              m_voxelArray->operator()(h, w, d) = j;
              break;
            }
          }
        }
      }
    }
  }

  StlVoxelGeometry(
      int nx, int ny, int nz, std::vector<stl_mesh::StlMesh *> meshes,
      Eigen::Matrix3f globalTransform = Eigen::Matrix3f::Identity())
      : VoxelGeometry(nx, ny, nz), m_meshes(meshes) {
    transform(globalTransform);
    setScale(Eigen::Vector3f(nx, ny, nz));
  }
};
