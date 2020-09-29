#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "StdUtils.hpp"
#include "UnitConverter.hpp"
#include "Vector3.hpp"
#include "VoxelArray.hpp"

/**
 * @brief Stores properites of a boundary condition
 *
 */
struct BoundaryCondition {
  //! The numerical ID associated to this boundary condition
  voxel_t m_id;
  //! Type of boundary condition
  VoxelType::Enum m_type;
  //! Temperature generated
  real m_temperature;
  //! Fluid velocity generated
  vector3<real> m_velocity;
  //! Plane normal of this boundary condition
  vector3<int> m_normal;
  //! Relative position of temperature condition (in voxel units)
  vector3<int> m_rel_pos;
  //! Thermal transfer time constant 1
  real m_tau1;
  //! Thermal transfer time constant 2
  real m_tau2;
  //! Fraction deciding initial thermal transfer
  real m_lambda;

  void setTemperature(const UnitConverter& uc, real temperature) {
    m_temperature = uc.Temp_to_lu(temperature);
  }

  void setFlow(const UnitConverter& uc, real flow, real area) {
    real velocityLu = fmaxf(0.0, uc.Q_to_Ulu(flow, area));
    vector3<real> nVelocity =
        vector3<real>(m_normal.x(), m_normal.y(), m_normal.z()).normalize();
    if (m_type == VoxelType::INLET_ZERO_GRADIENT) nVelocity = -nVelocity;
    m_velocity.x() = nVelocity.x() * velocityLu;
    m_velocity.y() = nVelocity.y() * velocityLu;
    m_velocity.z() = nVelocity.z() * velocityLu;
  }

  real getTemperature(const UnitConverter& uc) {
    return uc.luTemp_to_Temp(m_temperature);
  }

  real getFlow(const UnitConverter& uc, int areaLu) {
    return uc.Ulu_to_Q(m_velocity.norm(), areaLu);
  }

  /**
   * @brief Empty boundary condition
   *
   */
  CUDA_CALLABLE_MEMBER BoundaryCondition()
      : m_id(0),
        m_type(VoxelType::Enum::FLUID),
        m_temperature(NaN),
        m_velocity(vector3<real>(NaN, NaN, NaN)),
        m_normal(vector3<int>(0, 0, 0)),
        m_rel_pos(vector3<int>(0, 0, 0)),
        m_tau1(0),
        m_tau2(0),
        m_lambda(0) {}

  /**
   * @brief Copy constructor
   *
   * @param other Another bounary condition
   */
  explicit BoundaryCondition(const BoundaryCondition* other)
      : m_id(other->m_id),
        m_type(other->m_type),
        m_temperature(other->m_temperature),
        m_velocity(other->m_velocity),
        m_normal(other->m_normal),
        m_rel_pos(other->m_rel_pos),
        m_tau1(other->m_tau1),
        m_tau2(other->m_tau2),
        m_lambda(other->m_lambda) {}

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
                    vector3<real> velocity, vector3<int> normal,
                    vector3<int> rel_pos, real tau1, real tau2, real lambda)
      : m_id(id),
        m_type(type),
        m_temperature(temperature),
        m_velocity(velocity),
        m_normal(normal),
        m_rel_pos(rel_pos),
        m_tau1(tau1),
        m_tau2(tau2),
        m_lambda(lambda) {}
};

std::ostream& operator<<(std::ostream& os, VoxelType::Enum v);
std::ostream& operator<<(std::ostream& os, BoundaryCondition bc);
bool operator==(BoundaryCondition const& a, BoundaryCondition const& b);

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
  std::size_t operator()(const BoundaryCondition& bc,
                         const std::string& name = "") const {
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
    if (!std::isnan(bc.m_tau1)) ::hash_combine(&seed, bc.m_tau1);
    if (!std::isnan(bc.m_tau2)) ::hash_combine(&seed, bc.m_tau2);
    if (!std::isnan(bc.m_lambda)) ::hash_combine(&seed, bc.m_lambda);

    std::hash<std::string> strHash;
    ::hash_combine(&seed, strHash(name));
    return seed;
  }
};
}  // namespace std
