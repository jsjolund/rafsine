#pragma once

#include <QMutex>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using sim_clock_t = std::chrono::high_resolution_clock;
using sim_clock_t_timer_t = std::chrono::steady_clock;

std::ostream& operator<<(std::ostream& os, const sim_clock_t::time_point& tp);

class TimerCallback {
 private:
  std::chrono::duration<double> m_repeat;
  sim_clock_t::time_point m_timeout;
  bool m_paused;

 public:
  TimerCallback()
      : m_repeat(std::chrono::duration<double>(0)),
        m_timeout(sim_clock_t::from_time_t(0)),
        m_paused(false) {}

  virtual void run(uint64_t ticks, sim_clock_t::time_point simTime) = 0;

  virtual void reset() = 0;

  void pause(bool state) { m_paused = state; }

  bool isPaused() const { return m_paused; }

  void setRepeatTime(std::chrono::duration<double> sec) { m_repeat = sec; }

  void setRepeatTime(double sec) {
    m_repeat = std::chrono::duration<double>(sec);
  }

  std::chrono::duration<double> getRepeatTime() { return m_repeat; }

  bool isRepeating() { return m_repeat > std::chrono::duration<double>(0); }

  void setTimeout(sim_clock_t::time_point tp) { m_timeout = tp; }

  void setTimeout(unsigned int t) { m_timeout = sim_clock_t::from_time_t(t); }

  sim_clock_t::time_point getTimeout() { return m_timeout; }

  TimerCallback& operator=(const TimerCallback& other) {
    m_repeat = other.m_repeat;
    m_timeout = other.m_timeout;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const TimerCallback& timer) {
    os << "Timeout: " << timer.m_timeout;
    return os;
  }
};

class BasicTimer {
 protected:
  QMutex m_mutex;
  //! Length of one time step in simulated seconds
  double m_timeStep;
  //! Simulated time
  sim_clock_t::time_point m_simTime;
  //! Number of simulation updates
  uint64_t m_ticks;
  //! Initial time when resetting timer
  unsigned int m_initialTime;
  //! Simulation timer callbacks
  std::vector<std::shared_ptr<TimerCallback>> m_timerCallbacks;

 public:
  BasicTimer(double timeStep, unsigned int initialTime)
      : m_timeStep(timeStep),
        m_ticks(0),
        m_initialTime(initialTime),
        m_simTime(sim_clock_t::from_time_t(initialTime)) {}

  ~BasicTimer() { m_timerCallbacks.clear(); }

  void setStartTime(int initialTime) {
    m_initialTime = initialTime;
    m_simTime = sim_clock_t::from_time_t(initialTime);
  }

  sim_clock_t::time_point getTime() { return m_simTime; }

  void reset() {
    m_ticks = 0;
    m_simTime = sim_clock_t::from_time_t(m_initialTime);
  }

  void addTimerCallback(std::shared_ptr<TimerCallback> cb);

  void tick();

  friend std::ostream& operator<<(std::ostream& os, const BasicTimer& timer) {
    os << timer.m_simTime;
    return os;
  }
};
