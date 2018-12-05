#pragma once

#include <math.h>

#include "BoundaryCondition.hpp"

/**
 * @brief Converts between real units and discrete LBM units
 *
 */
class UnitConverter {
 public:
  //! Reference length in meters
  real m_ref_L_phys;
  //! Reference length in number of nodes
  real m_ref_L_lbm;
  //! Reference speed in meter/second
  real m_ref_U_phys;
  //! Reference speed in lattice units (linked to the Mach number)
  real m_ref_U_lbm;
  //! Temperature conversion factor
  real m_C_Temp;
  //! Reference temperature for Boussinesq in degres Celsius
  real m_T0_phys;
  //! Reference temperature in lattice units
  real m_T0_lbm;

  UnitConverter(real ref_L_phys, real ref_L_lbm, real ref_U_phys,
                real ref_U_lbm, real C_Temp, real T0_phys, real T0_lbm)
      : m_ref_L_phys(ref_L_phys),
        m_ref_L_lbm(ref_L_lbm),
        m_ref_U_phys(ref_U_phys),
        m_ref_U_lbm(ref_U_lbm),
        m_C_Temp(C_Temp),
        m_T0_phys(T0_phys),
        m_T0_lbm(T0_lbm) {}

  UnitConverter()
      : m_ref_L_phys(0),
        m_ref_L_lbm(0),
        m_ref_U_phys(0),
        m_ref_U_lbm(0),
        m_C_Temp(0),
        m_T0_phys(0),
        m_T0_lbm(0) {}

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

  //! Length conversion factor
  real C_L() { return m_ref_L_phys / (m_ref_L_lbm - 1); }
  //! Speed conversion factor
  real C_U() { return m_ref_U_phys / m_ref_U_lbm; }
  //! Time conversion factor
  real C_T() { return C_L() / C_U(); }

  void set(real ref_L_phys, real ref_L_lbm, real ref_U_phys, real ref_U_lbm,
           real C_Temp, real T0_phys, real T0_lbm) {
    m_ref_L_phys = ref_L_phys;
    m_ref_L_lbm = ref_L_lbm;
    m_ref_U_phys = ref_U_phys;
    m_ref_U_lbm = ref_U_lbm;
    m_C_Temp = C_Temp;
    m_T0_phys = T0_phys;
    m_T0_lbm = T0_lbm;
  }

  /**
   * @brief Round a real number to lattice units
   *
   * @param number
   * @return int
   */
  int round(real number) { return floor(number + 0.5); }

  /**
   * @brief Convert a distance in meters to a number of node (lattice unit)
   *
   * @param L_phys
   * @return int
   */
  int m_to_lu(real L_phys) { return this->round(L_phys / C_L()); }

  /**
   * @brief Convert a distance in meters to a number of node (lattice unit)
   *
   * @param L_phys
   * @param L_lbm
   */
  void m_to_lu(vec3<real> &L_phys, vec3<int> &L_lbm) {
    L_lbm.x = m_to_lu(L_phys.x);
    L_lbm.y = m_to_lu(L_phys.y);
    L_lbm.z = m_to_lu(L_phys.z);
  }

  /**
   * @brief Convert a position in real units to a node-based
   * position in lua (shifted by 1 compared to C++)
   *
   * @param L_phys
   * @return int
   */
  int m_to_LUA(real L_phys) { return m_to_lu(L_phys) + 1; }

  /**
   * @brief  Convert a position in real units to a node-based
   * position in lua (shifted by 1 compared to C++)
   *
   * @param L_phys
   * @param L_lbm
   */
  void m_to_LUA(vec3<real> &L_phys, vec3<int> &L_lbm) {
    L_lbm.x = m_to_lu(L_phys.x) + 1;
    L_lbm.y = m_to_lu(L_phys.y) + 1;
    L_lbm.z = m_to_lu(L_phys.z) + 1;
  }

  /**
   * @brief Convert a speed in meters/second to lattice units
   *
   * @param U_phys
   * @return real
   */
  real ms_to_lu(real U_phys) { return U_phys / C_U(); }

  /**
   * @brief Convert a volume flow rate in meters^3 / second to a
   * velocity in lattice unit
   *
   * @param Q_phys
   * @param A_phys
   * @return real
   */
  real Q_to_Ulu(real Q_phys, real A_phys) { return Q_phys / (C_U() * A_phys); }

  /**
   * @brief Convert velocity and area in lattice units to flow rate
   * in meters^3 / second
   *
   * @param Ulu
   * @param A_lu
   * @return real
   */
  real Ulu_to_Q(real Ulu, int A_lu) {
    return Ulu * C_U() * A_lu * C_L() * C_L();
  }

  /**
   * @brief Convert the kinematic viscosity in meters^2 / second to lattice
   * units
   *
   * @param Nu_phys
   * @return real
   */
  real Nu_to_lu(real Nu_phys) { return Nu_phys / (C_U() * C_L()); }

  /**
   * @brief Compute the relaxation time from the kinematic viscosity in meters^2
   * / second
   *
   * @param Nu_phys
   * @return real
   */
  real Nu_to_tau(real Nu_phys) { return 0.5 + 3 * Nu_to_lu(Nu_phys); }

  /**
   * @brief Compute the time convertion factor, i.e. the duration of one
   * time-step ( in seconds)
   *
   * @param nbr_iter
   * @return real
   */
  real N_to_s(real nbr_iter) { return C_T() * nbr_iter; }

  /**
   * @brief Convert seconds to number of time-steps
   *
   * @param seconds
   * @return int
   */
  int s_to_N(real seconds) { return this->round(seconds / C_T()); }

  /**
   * @brief Convert physical temperature in Celsius to lbm temperature in
   * lattice units
   *
   * @param Temp_phys
   * @return real
   */
  real Temp_to_lu(real Temp_phys) {
    return m_T0_lbm + 1 / m_C_Temp * (Temp_phys - m_T0_phys);
  }

  /**
   * @brief Convert temperature in lattice units to physical temperature in
   * Celsius
   *
   * @param Temp_lu
   * @return real
   */
  real luTemp_to_Temp(real Temp_lu) {
    return (Temp_lu - m_T0_lbm) * m_C_Temp + m_T0_phys;
  }

  /**
   * @brief Convert g*Betta, i.e., gravity acceleration * coefficient of thermal
   * expansion to lattice units
   *
   * @param gBetta_phys
   * @return real
   */
  real gBetta_to_lu(real gBetta_phys) {
    return gBetta_phys * C_T() * C_T() * m_C_Temp / C_L();
  }
};
