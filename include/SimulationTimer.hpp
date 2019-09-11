#pragma once

#include <QMutex>

#include <osg/Timer>

#include <stdint.h>
#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#define SIM_STATS_UPDATE_PERIOD 1.0

int timeval_subtract(const struct timeval &x, const struct timeval &y,
                     struct timeval *result = NULL);
void timeval_add(const timeval &a, const timeval &b, timeval *result);
void timeval_add_seconds(const timeval &t, const double seconds,
                         timeval *result);

/**
 * @brief This timer keeps track of the amount of simulated time, statistics
 * which depend on this, and functionality for executing tasks which depend on
 * simulation time.
 *
 */
class SimulationTimerCallback {
 public:
  timeval m_repeat;
  timeval m_timeout;
  SimulationTimerCallback()
      : m_timeout({.tv_sec = 0, .tv_usec = 0}),
        m_repeat({.tv_sec = 0, .tv_usec = 0}) {}
  virtual ~SimulationTimerCallback() {}
  virtual void run(uint64_t ticks) = 0;

  void setTimeout(timeval t) { m_timeout = t; }
  void setTimeout(double sec) {
    m_timeout.tv_sec = static_cast<int>(sec);
    m_timeout.tv_usec = static_cast<int>((sec - m_timeout.tv_sec) * 1000000);
  }
  void setRepeatTime(timeval t) { m_repeat = t; }
  void setRepeatTime(double sec) {
    m_repeat.tv_sec = static_cast<int>(sec);
    m_repeat.tv_usec = static_cast<int>((sec - m_repeat.tv_sec) * 1000000);
  }
  bool isRepeating() { return m_repeat.tv_sec > 0 || m_repeat.tv_usec > 0; }
};

class SimulationTimer {
 private:
  // Size of the lattice
  unsigned int m_latticeSize;
  // Tracks number of simulation updates for the purpose of updating stats,
  // automatically reset
  unsigned int m_latticeUpdateCounter;
  // Tracks total number of simulation updates
  uint64_t m_ticks;
  // Seconds simulated per update
  double m_secSimPerUpdate;

  // Tracks when to update statistics
  osg::Timer m_statsTimer;
  // Total simulated time in seconds
  timeval m_simTime;
  // Current million lattice updates per seconds
  unsigned int m_currentMlups;
  uint64_t m_totalMlups;
  unsigned int m_totalMlupsUpdates;

  // Current number of lattice updates per seconds
  unsigned int m_currentLups;
  // Current rate of simulated time to real time
  double m_realTimeRate;
  // Simulation timer callbacks
  std::vector<std::shared_ptr<SimulationTimerCallback>> m_timerCallbacks;

  QMutex m_mutex;

 public:
  inline int getMLUPS() { return m_currentMlups; }
  inline int getLUPS() { return m_currentLups; }
  inline double getRealTimeRate() { return m_realTimeRate; }
  inline timeval getSimulationTime() const { return m_simTime; }
  inline int getAverageMLUPS() { return m_totalMlups / m_totalMlupsUpdates; }

  SimulationTimer(unsigned int latticeSize, double secSimPerUpdate);
  void setSimulationTime(timeval newTime);
  void setSimulationTime(long newTime);
  void addSimulationTimer(std::shared_ptr<SimulationTimerCallback> cb);
  void tick();
  uint64_t getTicks() { return m_ticks; }
  void reset();
  ~SimulationTimer() { m_timerCallbacks.clear(); }
};

std::ostream &operator<<(std::ostream &os, const SimulationTimer &timer);
std::ostream &operator<<(std::ostream &os, const timeval &tval);
