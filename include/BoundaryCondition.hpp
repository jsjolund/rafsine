#pragma once

#include <string>
#include <vector>

#include "Primitives.hpp"
#include "VoxelArray.hpp"

/**
 * @brief Stores properites of a boundary condition
 *
 */
class BoundaryCondition {
 public:
  voxel m_id;  //!< The numerical ID associated to this boundary condition
  VoxelType::Enum m_type;  //!< Type of boundary condition
  real m_temperature;      //!< Temperature generated
  vec3<real> m_velocity;   //!< Fluid velocity generated
  vec3<int> m_normal;      //!< Plane normal of this boundary condition
  vec3<int> m_rel_pos;     //!<  Relative position of temperature condition (in
                           //!<  voxel units)

  /**
   * @brief Empty boundary condition
   *
   */
  BoundaryCondition()
      : m_id(0),
        m_type(VoxelType::Enum::FLUID),
        m_temperature(NaN),
        m_velocity(vec3<real>(NaN, NaN, NaN)),
        m_normal(vec3<int>(0, 0, 0)),
        m_rel_pos(vec3<int>(0, 0, 0)) {}

  /**
   * @brief Copy constructor
   *
   * @param other Another bounary condition
   */
  explicit BoundaryCondition(BoundaryCondition *other)
      : m_id(other->m_id),
        m_type(other->m_type),
        m_temperature(other->m_temperature),
        m_velocity(other->m_velocity),
        m_normal(other->m_normal),
        m_rel_pos(other->m_rel_pos) {}

  /**
   * @brief Construct a new Boundary Condition object
   *
   * @param id  The numerical ID associated to this boundary condition
   * @param type Type of boundary condition
   * @param temperature Temperature generated
   * @param velocity  Fluid velocity generated
   * @param normal  Plane normal of this boundary condition
   * @param rel_pos Relative position of temperature condition (in voxel units)
   */
  BoundaryCondition(int id, VoxelType::Enum type, real temperature,
                    vec3<real> velocity, vec3<int> normal, vec3<int> rel_pos)
      : m_id(id),
        m_type(type),
        m_temperature(temperature),
        m_velocity(velocity),
        m_normal(normal),
        m_rel_pos(rel_pos) {}
};

std::ostream &operator<<(std::ostream &os, VoxelType::Enum v);
std::ostream &operator<<(std::ostream &os, BoundaryCondition bc);
bool operator==(BoundaryCondition const &a, BoundaryCondition const &b);

namespace std {
template <>
/**
 * @brief Hashing function for a boundary condition used by the voxel geometry
 *
 */
struct hash<BoundaryCondition> {
  /**
   * @brief Hashing function for a boundary condition used by the voxel geometry
   *
   * @param bc Boundary condition to hash
   * @param name Optional name string
   * @return std::size_t The hash value
   */
  std::size_t operator()(const BoundaryCondition &bc,
                         const std::string &name = "") const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(seed, bc.m_type);
    ::hash_combine(seed, bc.m_normal.x);
    ::hash_combine(seed, bc.m_normal.y);
    ::hash_combine(seed, bc.m_normal.z);

    // Avoids issue with +/- NaN
    if (!std::isnan(bc.m_velocity.x)) ::hash_combine(seed, bc.m_velocity.x);
    if (!std::isnan(bc.m_velocity.y)) ::hash_combine(seed, bc.m_velocity.y);
    if (!std::isnan(bc.m_velocity.z)) ::hash_combine(seed, bc.m_velocity.z);
    if (!std::isnan(bc.m_temperature)) ::hash_combine(seed, bc.m_temperature);
    if (!std::isnan(bc.m_rel_pos.x)) ::hash_combine(seed, bc.m_rel_pos.x);
    if (!std::isnan(bc.m_rel_pos.y)) ::hash_combine(seed, bc.m_rel_pos.y);
    if (!std::isnan(bc.m_rel_pos.z)) ::hash_combine(seed, bc.m_rel_pos.z);

    std::hash<std::string> strHash;
    ::hash_combine(seed, strHash(name));
    return seed;
  }
};
}  // namespace std
