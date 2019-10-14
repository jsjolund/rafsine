#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "Eigen/Geometry"

#include "StdUtils.hpp"
#include "UnitConverter.hpp"
#include "VoxelArray.hpp"

/**
 * @brief Stores properites of a boundary condition
 *
 */
struct BoundaryCondition {
  voxel_t m_id;  //!< The numerical ID associated to this boundary condition
  VoxelType::Enum m_type;      //!< Type of boundary condition
  real m_temperature;          //!< Temperature generated
  Eigen::Vector3f m_velocity;  //!< Fluid velocity generated
  Eigen::Vector3i m_normal;    //!< Plane normal of this boundary condition
  Eigen::Vector3i m_rel_pos;   //!< Relative position of temperature condition
                               //!< (in voxel units)

  void setTemperature(const UnitConverter &uc, real temperature) {
    m_temperature = uc.Temp_to_lu(temperature);
  }

  void setFlow(const UnitConverter &uc, real flow, real area) {
    real velocityLu = max(0.0, uc.Q_to_Ulu(flow, area));
    Eigen::Vector3f nVelocity =
        Eigen::Vector3f(m_normal.x(), m_normal.y(), m_normal.z()).normalized();
    if (m_type == VoxelType::INLET_ZERO_GRADIENT) nVelocity = -nVelocity;
    m_velocity.x() = nVelocity.x() * velocityLu;
    m_velocity.y() = nVelocity.y() * velocityLu;
    m_velocity.z() = nVelocity.z() * velocityLu;
  }

  real getTemperature(const UnitConverter &uc) {
    return uc.luTemp_to_Temp(m_temperature);
  }

  real getFlow(const UnitConverter &uc, int areaLu) {
    return uc.Ulu_to_Q(m_velocity.norm(), areaLu);
  }

  /**
   * @brief Empty boundary condition
   *
   */
  BoundaryCondition()
      : m_id(0),
        m_type(VoxelType::Enum::FLUID),
        m_temperature(NaN),
        m_velocity(Eigen::Vector3f(NaN, NaN, NaN)),
        m_normal(Eigen::Vector3i(0, 0, 0)),
        m_rel_pos(Eigen::Vector3i(0, 0, 0)) {}

  /**
   * @brief Copy constructor
   *
   * @param other Another bounary condition
   */
  explicit BoundaryCondition(const BoundaryCondition *other)
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
                    Eigen::Vector3f velocity, Eigen::Vector3i normal,
                    Eigen::Vector3i rel_pos)
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

typedef std::vector<BoundaryCondition> BoundaryConditions;

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
    ::hash_combine(&seed, bc.m_type, bc.m_normal.x(), bc.m_normal.y(),
                   bc.m_normal.z());

    // Avoids issue with +/- NaN
    if (!std::isnan(bc.m_velocity.x()))
      ::hash_combine(&seed, bc.m_velocity.x());
    if (!std::isnan(bc.m_velocity.y()))
      ::hash_combine(&seed, bc.m_velocity.y());
    if (!std::isnan(bc.m_velocity.z()))
      ::hash_combine(&seed, bc.m_velocity.z());
    if (!std::isnan(bc.m_temperature)) ::hash_combine(&seed, bc.m_temperature);
    if (!std::isnan(bc.m_rel_pos.x())) ::hash_combine(&seed, bc.m_rel_pos.x());
    if (!std::isnan(bc.m_rel_pos.y())) ::hash_combine(&seed, bc.m_rel_pos.y());
    if (!std::isnan(bc.m_rel_pos.z())) ::hash_combine(&seed, bc.m_rel_pos.z());

    std::hash<std::string> strHash;
    ::hash_combine(&seed, strHash(name));
    return seed;
  }
};
}  // namespace std
