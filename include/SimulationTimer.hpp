#pragma once

#include <QMutex>

#include <osg/Timer>

#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include <vector>

#define SIM_STATS_UPDATE_PERIOD 1.0

int timeval_subtract(struct timeval *result, struct timeval *x,
                     struct timeval *y);
void timeval_add(timeval *t, double seconds);

// TODO: This should be (unit) tested
class SimulationTimerCallback {
 public:
  const timeval m_timeout;
  explicit SimulationTimerCallback(timeval timeout) : m_timeout(timeout) {}
  virtual ~SimulationTimerCallback() {}
  virtual void run() = 0;
};

class SimulationTimer {
 private:
  // Size of the lattice
  unsigned int m_latticeSize;
  // Tracks number of simulation updates
  unsigned int m_latticeUpdateCounter;
  // Seconds simulated per update
  double m_secSimPerUpdate;

  // Tracks when to update statistics
  osg::Timer m_statsTimer;
  // Total simulated time in seconds
  timeval m_simTime;
  // Current million lattice updates per seconds
  int m_currentMlups;
  // Current number of lattice updates per seconds
  int m_currentLups;
  // Current rate of simulated time to real time
  double m_realTimeRate;
  // Simulation timer callbacks
  std::vector<SimulationTimerCallback *> m_timerCallbacks;

  QMutex m_mutex;

 public:
  inline int getMLUPS() { return m_currentMlups; }
  inline int getLUPS() { return m_currentLups; }
  inline double getRealTimeRate() { return m_realTimeRate; }
  inline timeval getSimulationTime() const { return m_simTime; }

  SimulationTimer(unsigned int latticeSize, double secSimPerUpdate);
  void setSimulationTime(timeval newTime);
  void addSimulationTimeout(SimulationTimerCallback *cb);
  void tick();
  void reset();
};

std::ostream &operator<<(std::ostream &os, const SimulationTimer &timer);
