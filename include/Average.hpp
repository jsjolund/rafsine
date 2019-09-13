#pragma once

#include <sys/time.h>
#include <vector>

#include "CudaUtils.hpp"
#include "UnitConverter.hpp"
#include "VoxelObject.hpp"

class LatticeAverage {
 public:
  const real m_luTemperature;
  const real m_luVelocityX;
  const real m_luVelocityY;
  const real m_luVelocityZ;

  LatticeAverage(real temperature, real velX, real velY, real velZ)
      : m_luTemperature(temperature),
        m_luVelocityX(velX),
        m_luVelocityY(velY),
        m_luVelocityZ(velZ) {}
};

class Average {
 private:
  real getTemperature(const UnitConverter &uc, real luTemperature) const {
    return uc.luTemp_to_Temp(luTemperature);
  }
  real getVelocity(const UnitConverter &uc, real luVelocityX, real luVelocityY,
                   real luVelocityZ) const {
    return uc.C_U() *
           sqrt(luVelocityX * luVelocityX + luVelocityY * luVelocityY +
                luVelocityZ * luVelocityZ);
  }
  real getFlow(const UnitConverter &uc, const VoxelVolume &vol,
               real luVelocityX, real luVelocityY, real luVelocityZ) const {
    return getVelocity(uc, luVelocityX, luVelocityY, luVelocityZ) *
           vol.getNumVoxels() * pow(uc.C_L(), vol.getRank());
  }

 public:
  const VoxelVolume m_volume;
  const real m_temperature;
  const real m_velocity;
  const real m_flow;

  Average(const UnitConverter &uc, const VoxelVolume &vol, LatticeAverage lAvg)
      : m_volume(vol),
        m_temperature(getTemperature(uc, lAvg.m_luTemperature)),
        m_velocity(getVelocity(uc, lAvg.m_luVelocityX, lAvg.m_luVelocityY,
                               lAvg.m_luVelocityZ)),
        m_flow(getFlow(uc, vol, lAvg.m_luVelocityX, lAvg.m_luVelocityY,
                       lAvg.m_luVelocityZ)) {}
};

struct AverageData {
  timeval time;
  std::vector<Average> rows;
};
