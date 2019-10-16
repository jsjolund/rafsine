#pragma once

#include <string>
#include <vector>
#include <chrono>

#include "CudaUtils.hpp"
#include "UnitConverter.hpp"
#include "VoxelObject.hpp"

class LatticeAverage {
 public:
  const real m_luTemperature;
  const real m_luVelocityX;
  const real m_luVelocityY;
  const real m_luVelocityZ;

  real getTemperature(const UnitConverter& uc) const {
    return uc.luTemp_to_Temp(m_luTemperature);
  }

  real getVelocity(const UnitConverter& uc) const {
    return uc.C_U() *
           sqrt(m_luVelocityX * m_luVelocityX + m_luVelocityY * m_luVelocityY +
                m_luVelocityZ * m_luVelocityZ);
  }

  real getFlow(const UnitConverter& uc, const VoxelVolume& vol) const {
    return getVelocity(uc) * vol.getNumVoxels() * pow(uc.C_L(), vol.getRank());
  }

  LatticeAverage(real temperature, real velX, real velY, real velZ)
      : m_luTemperature(temperature),
        m_luVelocityX(velX),
        m_luVelocityY(velY),
        m_luVelocityZ(velZ) {}
};

struct Average {
  std::string m_name;
  real m_temperature;
  real m_velocity;
  real m_flow;

  Average() : m_name(), m_temperature(0), m_velocity(0), m_flow(0) {}

  Average(const UnitConverter& uc,
          const VoxelVolume& vol,
          const LatticeAverage& lAvg)
      : m_name(vol.getName()),
        m_temperature(lAvg.getTemperature(uc)),
        m_velocity(lAvg.getVelocity(uc)),
        m_flow(lAvg.getFlow(uc, vol)) {}

  Average& operator=(const Average& other) {
    m_name = other.m_name;
    m_temperature = other.m_temperature;
    m_velocity = other.m_velocity;
    m_flow = other.m_flow;
    return *this;
  }
};

struct AverageData {
  std::chrono::system_clock::time_point m_time;
  std::vector<Average> m_measurements;
};
