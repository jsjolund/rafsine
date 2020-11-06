#pragma once

#include <QMutex>

#include "BasicTimer.hpp"

#define SIM_STATS_UPDATE_PERIOD 1.0

/**
 * @brief Calculates statistics about rates and performances based on simulated
 * time versus real time.
 */
class SimulationTimer : public BasicTimer {
 private:
  //! Size of the lattice
  size_t m_latticeSize;
  //! Current million lattice updates per seconds
  unsigned int m_currentMlups;
  //! Sum of MLUPS updates
  uint64_t m_totalMlups;
  //! Number of MLUPS updates
  unsigned int m_totalMlupsUpdates;
  //! Current number of lattice updates per seconds
  unsigned int m_currentLups;
  //! Current ratio of simulated time to real time
  double m_realTimeRate;
  // Tracks real time for updating simulation stats
  sim_clock_t_timer_t::time_point m_statsTimer;
  //! Number of ticks since last stats update
  uint64_t m_statsTicks;

 public:
  /**
   * @brief Construct a new Simulation Timer
   *
   * @param latticeSize Number of lattice sites
   * @param timeStep Number of real seconds per simulation time step
   * @param initialTime Initial time and date as UNIX time stamp
   */
  SimulationTimer(size_t latticeSize,
                  double timeStep,
                  unsigned int initialTime = 0)
      : BasicTimer(timeStep, initialTime),
        m_latticeSize(latticeSize),
        m_currentMlups(0),
        m_totalMlups(0),
        m_totalMlupsUpdates(0),
        m_currentLups(0),
        m_realTimeRate(0),
        m_statsTimer(sim_clock_t_timer_t::now()),
        m_statsTicks(0) {}

  /**
   * @return double Rate of simulated time to real time. If >1 simulation is
   * running faster than real time.
   */
  inline double getRealTimeRate() { return m_realTimeRate; }

  /**
   * @brief Million Lattice site Updates Per Second
   *
   * @return unsigned int
   */
  inline unsigned int getMLUPS() { return m_currentMlups; }

  /**
   * @brief Lattice Updates Per Second
   *
   * @return unsigned int
   */
  inline unsigned int getLUPS() { return m_currentLups; }

  /**
   * @brief Average Million Lattice site Updates Per Second since simulation
   * start
   *
   * @return unsigned int
   */
  inline unsigned int getAverageMLUPS() {
    return (m_totalMlupsUpdates > 0) ? m_totalMlups / m_totalMlupsUpdates : 0;
  }

  /**
   * @brief Resets all timer stats to zero
   */
  void reset() {
    BasicTimer::reset();
    m_currentMlups = 0;
    m_totalMlups = 0;
    m_totalMlupsUpdates = 0;
    m_currentLups = 0;
    m_realTimeRate = 0;
    m_statsTimer = sim_clock_t_timer_t::now();
    m_statsTicks = 0;
  }

  /**
   * @brief Increment the timer by tick after each simulation update
   */
  void tick() {
    BasicTimer::tick();
    m_statsTicks++;
    sim_clock_t_timer_t::time_point now = sim_clock_t_timer_t::now();
    sim_duration_t timeSpan =
        std::chrono::duration_cast<sim_duration_t>(now - m_statsTimer);
    double statsTimeDelta = timeSpan.count();

    if (statsTimeDelta >= SIM_STATS_UPDATE_PERIOD) {
      m_currentMlups = static_cast<int>(1e-6 * m_statsTicks * m_latticeSize /
                                        statsTimeDelta);
      m_currentLups = static_cast<int>(m_statsTicks / statsTimeDelta);
      m_totalMlups += m_currentMlups;
      m_totalMlupsUpdates++;
      m_realTimeRate = m_statsTicks * m_timeStep / statsTimeDelta;

      m_statsTicks = 0;
      m_statsTimer = now;
    }
  }
};
