#pragma once

#include <limits>
#include <vector>

#include <glm/vec3.hpp>

#include "StlMesh.hpp"

void getExtents(const std::vector<stl_mesh::StlMesh> &meshes, glm::vec3 *min,
                glm::vec3 *max) {
  min->x = std::numeric_limits<float>::max();
  min->y = std::numeric_limits<float>::max();
  min->z = std::numeric_limits<float>::max();

  max->x = std::numeric_limits<float>::min();
  max->y = std::numeric_limits<float>::min();
  max->z = std::numeric_limits<float>::min();

  for (stl_mesh::StlMesh mesh : meshes) {
    for (int i = 0; i < mesh.vertices.size(); i += 3) {
      float x = mesh.vertices.at(i + 0);
      float y = mesh.vertices.at(i + 1);
      float z = mesh.vertices.at(i + 2);

      if (x < min->x) min->x = x;
      if (x > max->x) max->x = x;
      if (y < min->y) min->y = y;
      if (y > max->y) max->y = y;
      if (z < min->z) min->z = z;
      if (z > max->z) max->z = z;
    }
  }
}
