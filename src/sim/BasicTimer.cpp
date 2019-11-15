#include "BasicTimer.hpp"

std::ostream& operator<<(std::ostream& os, const sim_clock_t::time_point& tp) {
  auto tpNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(tp);
  // get number of nanoseconds for the current second
  auto durNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   tpNs.time_since_epoch()) %
               1000000000;
  // convert to std::time_t in order to convert to std::tm (broken time)
  auto tpTime = sim_clock_t::to_time_t(tp);
  // convert to broken down time with nanoseconds
  os << std::put_time(std::localtime(&tpTime), "%d %b %Y %H:%M:%S") << '.'
     << std::setfill('0') << std::setw(9) << durNs.count();
  return os;
}

void BasicTimer::removeTimerCallback(std::shared_ptr<TimerCallback> cb) {
  auto itr = std::find(m_timerCallbacks.begin(), m_timerCallbacks.end(), cb);
  if (itr != m_timerCallbacks.end()) m_timerCallbacks.erase(itr);
}

void BasicTimer::addTimerCallback(std::shared_ptr<TimerCallback> cb) {
  auto itr = std::find(m_timerCallbacks.begin(), m_timerCallbacks.end(), cb);
  if (itr == m_timerCallbacks.end()) m_timerCallbacks.push_back(cb);
}

void BasicTimer::tick() {
  m_ticks++;
  sim_duration_t ct_d(m_timeStep);
  m_simTime += std::chrono::duration_cast<std::chrono::nanoseconds>(ct_d);

  // Check if any timers should trigger
  for (std::shared_ptr<TimerCallback> cb : m_timerCallbacks) {
    if (cb->getTimeout() <= m_simTime && !cb->isPaused()) {
      cb->run(m_ticks, m_simTime);

      if (cb->isRepeating()) {
        sim_duration_t overshoot = (cb->getTimeout() < m_simTime)
                                       ? sim_duration_t(0)
                                       : m_simTime - cb->getTimeout();
        sim_duration_t period = cb->getRepeatTime();
        sim_clock_t::time_point nextTimeout =
            m_simTime +
            std::chrono::duration_cast<std::chrono::nanoseconds>(period) -
            std::chrono::duration_cast<std::chrono::nanoseconds>(overshoot);
        cb->setTimeout(nextTimeout);

      } else {
        removeTimerCallback(cb);
      }
    }
  }
}
