#pragma once

#include "UnitConverter.hpp"

typedef float real;
typedef tsl::ordered_map<std::string, std::string> UserConstants;

class SimConstants
{
public:
  UnitConverter *uc;
  // Size in meters
  real mx, my, mz;
  // Kinematic viscosity of air
  real nu;
  // Thermal diffusivity
  real nuT;
  // Smagorinsky constant
  real C;
  // Thermal conductivity
  real k;
  // Prandtl number of air
  real Pr;
  // Turbulent Prandtl number
  real Pr_t;
  // Gravity * thermal expansion
  real gBetta;
  // Initial temperature
  real Tinit;
  // Reference temperature for Boussinesq
  real Tref;

  SimConstants(UnitConverter *uc)
      : uc(uc), mx(0), my(0), mz(0), nu(0),
        nuT(0), C(0), k(0), Pr(0),
        Pr_t(0), gBetta(0), Tinit(0), Tref(0){};

  int nx() { return uc->m_to_LUA(mx); }
  int ny() { return uc->m_to_LUA(my); }
  int nz() { return uc->m_to_LUA(mz); }
  real gBetta_to_lu() { return uc->gBetta_to_lu(gBetta); }
  real Tinit_to_lu() { return uc->Temp_to_lu(Tinit); }
  real Tref_to_lu() { return uc->Temp_to_lu(Tref); }
};