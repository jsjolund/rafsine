#pragma once

#include <bits/stdc++.h>
#include <chrono>
#include <string>
#include <vector>

#include "BasicTimer.hpp"
#include "CudaUtils.hpp"
#include "UnitConverter.hpp"
#include "VoxelObject.hpp"

/**
 * @brief Stores average temperature, velocity and volumetric flow
 */
struct Average {
  real temperature;
  real velocity;
  real flow;
};

/**
 * @brief Stores averages of a particular point in time
 */
struct AverageData {
  sim_clock_t::time_point m_time;
  std::vector<Average> m_measurements;
};

/**
 * @brief Stores multiple averages over time
 */
struct AverageMatrix {
  std::vector<std::string> m_columns;
  std::vector<AverageData> m_rows;
};

/**
 * @brief Stores unconverted averages as read from the lattice
 */
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
