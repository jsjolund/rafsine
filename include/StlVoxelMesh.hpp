// #pragma once

// #include <limits>
// #include <vector>

// #include "StlMesh.hpp"
// // #include "VoxelGeometry.hpp"

// // class StlVoxelMesh : public VoxelGeometry {
// class StlVoxelMesh {
//  private:
//   std::vector<stl_mesh::StlMesh> m_meshes;
//   // VoxelGeometry vox;

//  public:
//   void getExtents(glm::vec3 *min, glm::vec3 *max) {
//     min->x = std::numeric_limits<float>::max();
//     min->y = std::numeric_limits<float>::max();
//     min->z = std::numeric_limits<float>::max();
//     max->x = std::numeric_limits<float>::min();
//     max->y = std::numeric_limits<float>::min();
//     max->z = std::numeric_limits<float>::min();

//     for (stl_mesh::StlMesh mesh : m_meshes) {
//       for (int i = 0; i < mesh.vertices.size(); i += 3) {
//         float x = mesh.vertices.at(i + 0);
//         float y = mesh.vertices.at(i + 1);
//         float z = mesh.vertices.at(i + 2);

//         if (x < min->x) min->x = x;
//         if (y < min->y) min->y = y;
//         if (z < min->z) min->z = z;
//         if (x > max->x) max->x = x;
//         if (y > max->y) max->y = y;
//         if (z > max->z) max->z = z;
//       }
//     }
//   }

//   StlVoxelMesh(int nx, int ny, int nz)
//       // : vox(nx, ny, nz)
//       m_meshes(meshes) {}
// };