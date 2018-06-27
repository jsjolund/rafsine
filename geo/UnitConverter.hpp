#pragma once

#include <math.h>

#include "../sim/BoundaryCondition.hpp"

class UnitConverter
{
public:
  // reference length in meters
  const real ref_L_phys;
  // reference length in number of nodes
  const real ref_L_lbm;
  // reference speed in meter/second
  const real ref_U_phys;
  // reference speed in lattice units (linked to the Mach number)
  const real ref_U_lbm;
  // temperature conversion factor
  const real C_Temp;
  // reference temperature for Boussinesq in degres Celsius
  const real T0_phys;
  const real T0_lbm;

  // --[[explanations on the lenght convertion factor:
  // --  designed for C/C++ index standard
  // --     [0; ...; N-1] --> total of N nodes
  // --     Xmin <-> 0
  // --     Xmax <-> N-1
  // --  C/C++ memory is allocated with array[N]
  // --                                 array[m_to_lu(Xmax)+1]
  // --
  // --  other possibilty: C_L = L_m / L_n
  // --                    [0; ...;N] --> total N+1 nodes
  // --                    Xmin <-> 0
  // --                    Xmax <-> N
  // --                    memory allocated with array[N+1]
  // --                                          array[m_to_lu(Xmax)+1]
  // --]]
  // length convertion factor
  const real C_L;
  // speed conversion factor
  const real C_U;
  // time conversion factor
  const real C_T;

  UnitConverter(
      real ref_L_phys,
      real ref_L_lbm,
      real ref_U_phys,
      real ref_U_lbm,
      real C_Temp,
      real T0_phys,
      real T0_lbm) : ref_L_phys(ref_L_phys),
                     ref_L_lbm(ref_L_lbm),
                     ref_U_phys(ref_U_phys),
                     ref_U_lbm(ref_U_lbm),
                     C_Temp(C_Temp),
                     T0_phys(T0_phys),
                     T0_lbm(T0_lbm),
                     C_L(ref_L_phys / (ref_L_lbm - 1)),
                     C_U(ref_U_phys / ref_U_lbm),
                     C_T(C_L / C_U)
  {
  }

  int round(real number)
  {
    return floor(number + 0.5);
  }

  // convert a distance in meters to a number of node (lattice unit)
  int m_to_lu(real L_phys)
  {
    return this->round(L_phys / C_L);
  }

  // convert a distance in meters to a number of node (lattice unit)
  void m_to_lu(vec3<real> &L_phys, vec3<int> &L_lbm)
  {
    L_lbm.x = m_to_lu(L_phys.x);
    L_lbm.y = m_to_lu(L_phys.y);
    L_lbm.z = m_to_lu(L_phys.z);
  }

  // function to convert a position in real units
  // to a node-based position in lua
  // (shifted by 1 compared to C++)
  int m_to_LUA(real L_phys)
  {
    return m_to_lu(L_phys) + 1;
  }

  void m_to_LUA(vec3<real> &L_phys, vec3<int> &L_lbm)
  {
    L_lbm.x = m_to_lu(L_phys.x) + 1;
    L_lbm.y = m_to_lu(L_phys.y) + 1;
    L_lbm.z = m_to_lu(L_phys.z) + 1;
  }

  // function to convert a speed in meters/second to lattice units
  real ms_to_lu(real U_phys)
  {
    return U_phys / C_U;
  }

  // function to convert a volume flow rate in meters^3 / second
  //  to a velocity in lattice unit
  real Q_to_Ulu(real Q_phys, real A_phys)
  {
    return Q_phys / (C_U * A_phys);
  }

  // function to convert the kinematic viscosity in meters^2 / second to lattice units
  real Nu_to_lu(real Nu_phys)
  {
    return Nu_phys / (C_U * C_L);
  }

  // function to compute the relaxation time from the kinematic viscosity in meters^2 / second
  real Nu_to_tau(real Nu_phys)
  {
    return 0.5 + 3 * Nu_to_lu(Nu_phys);
  }

  // function to compute the time convertion factor, i.e. the duration of one time-step ( in seconds)
  real N_to_s(real nbr_iter)
  {
    return C_T * nbr_iter;
  }

  // convert seconds to number of time-steps
  int s_to_N(real seconds)
  {
    return this->round(seconds / C_T);
  }

  // convert physical temperature in Celsius to lbm temperature in lattice units
  real Temp_to_lu(real Temp_phys)
  {
    return T0_lbm + 1 / C_Temp * (Temp_phys - T0_phys);
  }

  // convert g*Betta, i.e., gravity acceleration * coefficient of thermal expansion to lattice units
  real gBetta_to_lu(real gBetta_phys)
  {
    return gBetta_phys * C_T * C_T * C_Temp / C_L;
  }
};
