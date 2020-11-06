#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define DATETIME_FMT "%Y-%m-%d %H:%M:%S"

using sim_clock_t = std::chrono::high_resolution_clock;
using sim_clock_t_timer_t = std::chrono::steady_clock;
using sim_duration_t = std::chrono::duration<double>;

std::ostream& operator<<(std::ostream& os, const sim_clock_t::time_point& tp);

/**
 * @brief Execute a callback function at a specific point in time in the
 * simulation domain, such as averaging measurements.
 */
class TimerCallback {
 private:
  sim_duration_t m_repeat;
  sim_clock_t::time_point m_timeout;
  bool m_paused;

 public:
  /**
   * @brief Construct a new empty timer
   */
  TimerCallback()
      : m_repeat(sim_duration_t(0)),
        m_timeout(sim_clock_t::from_time_t(0)),
        m_paused(false) {}

  /**
   * @brief Callback for child class to run when timer is activated
   *
   * @param ticks Current number of total simulation ticks
   * @param simTime Current time in the simulation domain
   */
  virtual void run(uint64_t ticks, sim_clock_t::time_point simTime) = 0;

  /**
   * @brief Callback to reset the state of the child class when simulation is
   * restarted
   */
  virtual void reset() = 0;

  /**
   * @brief Pause the callback function
   *
   * @param state
   */
  void pause(bool state) { m_paused = state; }

  /**
   * @return true When simulation is paused
   * @return false When simulation is running
   */
  bool isPaused() const { return m_paused; }

  /**
   * @brief Set duration of repeated timer callbacks using duration structure
   *
   * @param sec
   */
  void setRepeatTime(sim_duration_t sec) { m_repeat = sec; }

  /**
   * @brief Set duration of repeated timer callbacks using seconds as double
   *
   * @param sec
   */
  void setRepeatTime(double sec) { m_repeat = sim_duration_t(sec); }

  /**
   * @brief Get the time interval between repeated timer activations
   *
   * @return sim_duration_t
   */
  sim_duration_t getRepeatTime() { return m_repeat; }

  /**
   * @return true If the timer is repeating
   * @return false If the timer is one-off
   */
  bool isRepeating() { return m_repeat > sim_duration_t(0); }

  /**
   * @brief Sets one-off timeout (non-repeating timer)
   *
   * @param tp Point in simulation time
   */
  void setTimeout(sim_clock_t::time_point tp) { m_timeout = tp; }

  /**
   * @brief Set one-off timeout using time_t format (non-repeating timer)
   *
   * @param t
   */
  void setTimeout(unsigned int t) { m_timeout = sim_clock_t::from_time_t(t); }

  /**
   * @brief Get the timeout as time point structure
   *
   * @return sim_clock_t::time_point
   */
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

/**
 * @brief Keeps track of current simulation time and callbacks to run at
 * different points in time
 */
class BasicTimer {
 protected:
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
        m_simTime(sim_clock_t::from_time_t(initialTime)),
        m_ticks(0),
        m_initialTime(initialTime),
        m_timerCallbacks() {}

  ~BasicTimer() { m_timerCallbacks.clear(); }

  void setStartTime(std::time_t initialTime) {
    m_initialTime = initialTime;
    m_simTime = sim_clock_t::from_time_t(initialTime);
  }

  sim_clock_t::time_point getTime() { return m_simTime; }

  void reset() {
    m_ticks = 0;
    m_simTime = sim_clock_t::from_time_t(m_initialTime);
  }

  void addTimerCallback(std::shared_ptr<TimerCallback> cb);
  void removeTimerCallback(std::shared_ptr<TimerCallback> cb);

  void tick();

  friend std::ostream& operator<<(std::ostream& os, const BasicTimer& timer) {
    os << timer.m_simTime;
    return os;
  }

  static std::time_t parseDatetime(std::string datetime) {
    std::tm tm = {};
    std::stringstream ss(datetime);
    ss >> std::get_time(&tm, DATETIME_FMT);
    return std::mktime(&tm) + timezone;
  }
};
