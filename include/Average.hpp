#pragma once

#include "CudaUtils.hpp"

class Average {
 public:
  real m_temperature;
  real m_velocityX;
  real m_velocityY;
  real m_velocityZ;
  Average()
      : m_temperature(0), m_velocityX(0), m_velocityY(0), m_velocityZ(0) {}
};
