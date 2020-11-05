#pragma once

#include <math.h>

#include "CudaUtils.hpp"
#include "Vector3.hpp"

/**
 * @brief Converts between real units and discrete LBM units
 *
 */
class UnitConverter {
 public:
  //! Reference length in meters
  real_t m_ref_L_phys;
  //! Reference length in number of nodes
  real_t m_ref_L_lbm;
  //! Reference speed in meter/second
  real_t m_ref_U_phys;
  //! Reference speed in lattice units (linked to the Mach number)
  real_t m_ref_U_lbm;
  //! Temperature conversion factor
  real_t m_C_Temp;
  //! Reference temperature for Boussinesq in degres Celsius
  real_t m_T0_phys;
  //! Reference temperature in lattice units
  real_t m_T0_lbm;

  UnitConverter(real_t ref_L_phys,
                real_t ref_L_lbm,
                real_t ref_U_phys,
                real_t ref_U_lbm,
                real_t C_Temp,
                real_t T0_phys,
                real_t T0_lbm)
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
  real_t C_L() const { return m_ref_L_phys / (m_ref_L_lbm - 1); }
  //! Speed conversion factor
  real_t C_U() const { return m_ref_U_phys / m_ref_U_lbm; }
  //! Time conversion factor
  real_t C_T() const { return C_L() / C_U(); }

  void set(real_t ref_L_phys,
           real_t ref_L_lbm,
           real_t ref_U_phys,
           real_t ref_U_lbm,
           real_t C_Temp,
           real_t T0_phys,
           real_t T0_lbm) {
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
  inline int round(real_t number) const { return floor(number + 0.5); }

  /**
   * @brief Convert a distance in meters to a number of node (lattice unit)
   *
   * @param L_phys
   * @return int
   */
  inline int m_to_lu(real_t L_phys) const { return this->round(L_phys / C_L()); }

  /**
   * @brief Convert a position in real units to a node-based
   * position in lua (shifted by 1 compared to C++)
   *
   * @param L_phys
   * @return int
   */
  inline int m_to_LUA(real_t L_phys) const { return m_to_lu(L_phys) + 1; }

  /**
   * @brief Convert a distance in meters to a number of node (lattice unit)
   *
   * @param L_phys
   * @param L_lbm
   */
  Vector3<int> m_to_lu_vec(Vector3<real_t> L_phys) const {
    Vector3<int> L_lbm;
    L_lbm.x() = m_to_lu(L_phys.x());
    L_lbm.y() = m_to_lu(L_phys.y());
    L_lbm.z() = m_to_lu(L_phys.z());
    return L_lbm;
  }

  /**
   * @brief  Convert a position in real units to a node-based
   * position in lua (shifted by 1 compared to C++)
   *
   * @param L_phys
   * @param L_lbm
   */
  Vector3<int> m_to_LUA_vec(Vector3<real_t> L_phys) const {
    Vector3<int> L_lbm;
    L_lbm.x() = m_to_lu(L_phys.x()) + 1;
    L_lbm.y() = m_to_lu(L_phys.y()) + 1;
    L_lbm.z() = m_to_lu(L_phys.z()) + 1;
    return L_lbm;
  }

  /**
   * @brief Convert a speed in meters/second to lattice units
   *
   * @param U_phys
   * @return real_t
   */
  inline real_t ms_to_lu(real_t U_phys) const { return U_phys / C_U(); }

  /**
   * @brief Convert a volume flow rate in meters^3 / second to a
   * velocity in lattice unit
   *
   * @param Q_phys
   * @param A_phys
   * @return real_t
   */
  inline real_t Q_to_Ulu(real_t Q_phys, real_t A_phys) const {
    return Q_phys / (C_U() * A_phys);
  }

  /**
   * @brief Convert velocity and area in lattice units to flow rate
   * in meters^3 / second
   *
   * @param Ulu
   * @param A_lu
   * @return real_t
   */
  inline real_t Ulu_to_Q(real_t Ulu, int A_lu) const {
    return Ulu * C_U() * A_lu * C_L() * C_L();
  }

  /**
   * @brief Convert the kinematic viscosity in meters^2 / second to lattice
   * units
   *
   * @param Nu_phys
   * @return real_t
   */
  inline real_t Nu_to_lu(real_t Nu_phys) const { return Nu_phys / (C_U() * C_L()); }

  /**
   * @brief Compute the relaxation time from the kinematic viscosity in meters^2
   * / second
   *
   * @param Nu_phys
   * @return real_t
   */
  inline real_t Nu_to_tau(real_t Nu_phys) const {
    return 0.5 + 3 * Nu_to_lu(Nu_phys);
  }

  /**
   * @brief Compute the time conversion factor, i.e. the duration of one
   * time-step (in seconds of simulated time)
   *
   * @param nbr_iter
   * @return real_t
   */
  inline real_t N_to_s(int nbr_iter) const { return C_T() * nbr_iter; }

  /**
   * @brief Convert seconds to number of time-steps (rounded up)
   *
   * @param seconds
   * @return int
   */
  inline unsigned int s_to_N(real_t seconds) const {
    return ceil(seconds / C_T());
  }

  /**
   * @brief Convert physical temperature in Celsius to lbm temperature in
   * lattice units
   *
   * @param Temp_phys
   * @return real_t
   */
  inline real_t Temp_to_lu(real_t Temp_phys) const {
    return m_T0_lbm + 1 / m_C_Temp * (Temp_phys - m_T0_phys);
  }

  /**
   * @brief Convert temperature in lattice units to physical temperature in
   * Celsius
   *
   * @param Temp_lu
   * @return real_t
   */
  inline real_t luTemp_to_Temp(real_t Temp_lu) const {
    return (Temp_lu - m_T0_lbm) * m_C_Temp + m_T0_phys;
  }

  /**
   * @brief Convert g*Betta, i.e., gravity acceleration * coefficient of thermal
   * expansion to lattice units
   *
   * @param gBetta_phys
   * @return real_t
   */
  inline real_t gBetta_to_lu(real_t gBetta_phys) const {
    return gBetta_phys * C_T() * C_T() * m_C_Temp / C_L();
  }
};
