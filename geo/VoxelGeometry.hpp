#pragma once

// #define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
// #include <glm/gtx/norm.hpp>
#include <string>
#include <unordered_map>
#include "../sim/BoundaryCondition.hpp"

using glm::ivec3;
using glm::vec3;
using std::string;

template <class T>
inline void hash_combine(std::size_t &seed, const T &v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

enum NodeType
{
  OVERWRITE,
  INTERSECT,
  FILL
};

class NodeUnit
{
public:
  string name;
  NodeType mode;
  VoxelType typeBC;
  real temperature, velocity;

  bool operator==(const NodeUnit &other) const
  {
    return (typeBC == other.typeBC &&
            temperature == other.temperature &&
            velocity == other.velocity);
  }
};

class VoxelUnit : public NodeUnit
{
public:
  ivec3 origin, dir1, dir2, normal;
  bool operator==(const VoxelUnit &other) const
  {
    return (
        NodeUnit::operator==(other) &&
        origin == other.origin &&
        dir1 == other.dir1 &&
        dir2 == other.dir2 &&
        normal == other.normal);
  }
};

namespace std
{
template <>
struct hash<VoxelUnit>
{
  std::size_t operator()(const VoxelUnit &k) const
  {
    using std::hash;
    using std::size_t;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:
    size_t seed = 0;
    ::hash_combine(seed, k.typeBC);
    ::hash_combine(seed, k.temperature);
    ::hash_combine(seed, k.velocity);
    ::hash_combine(seed, k.origin.x);
    ::hash_combine(seed, k.origin.y);
    ::hash_combine(seed, k.origin.z);
    ::hash_combine(seed, k.dir1.x);
    ::hash_combine(seed, k.dir1.y);
    ::hash_combine(seed, k.dir1.z);
    ::hash_combine(seed, k.dir2.x);
    ::hash_combine(seed, k.dir2.y);
    ::hash_combine(seed, k.dir2.z);
    ::hash_combine(seed, k.normal.x);
    ::hash_combine(seed, k.normal.y);
    ::hash_combine(seed, k.normal.z);
    return seed;
  }
};
}

class RealUnit : public NodeUnit
{
public:
  vec3 origin, dir1, dir2, normal;
  bool operator==(const RealUnit &other) const
  {
    return (
        NodeUnit::operator==(other) &&
        origin == other.origin &&
        dir1 == other.dir1 &&
        dir2 == other.dir2 &&
        normal == other.normal);
  }
};

class VoxelGeometry
{
private:
  int nx, ny, nz;
  int ***data;
  int newtype = 1;
  std::unordered_map<VoxelUnit,int> types;

public:
  void inline set(int x, int y, int z, int value) { *(&data)[x][y][z] = value; }
  int inline get(int x, int y, int z) { return *(&data)[x][y][z]; }
  int inline get(ivec3 v) { return *(&data)[v.x][v.y][v.z]; }

  int inline addQuadBCNode(VoxelUnit node)
  {
    int l1 = int(sqrt(node.dir1.x * node.dir1.x) + sqrt(node.dir1.y * node.dir1.y) + sqrt(node.dir1.z * node.dir1.z));
    int l2 = int(sqrt(node.dir2.x * node.dir2.x) + sqrt(node.dir2.y * node.dir2.y) + sqrt(node.dir2.z * node.dir2.z));
    ivec3 dir1 = ivec3(sgn(node.dir1.x),sgn(node.dir1.y),sgn(node.dir1.z));
    ivec3 dir2 = ivec3(sgn(node.dir2.x),sgn(node.dir2.y),sgn(node.dir2.z));
    for (int i = 0; i < l1; i++)
      for (int j = 0; j < l2; j++)
      {
        vec3 p = node.origin + i * dir1 + j * dir2;
        if (get(p) != EMPTY && get(p) != FLUID)
        {
        }
        else
        {
        }
      }
    return 0;
  }

  ~VoxelGeometry() { delete data; }
  VoxelGeometry(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz)
  {
    data = new int **[nx * ny * nz];
    for (int x = 0; x < nx; x++)
      for (int y = 0; y < ny; y++)
        for (int z = 0; z < nz; z++)
          set(x, y, z, 0);
  }
};