#include "BasicTimer.hpp"

std::ostream& operator<<(std::ostream& os, const sim_clock_t::time_point& tp) {
  auto tpNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(tp);
  // get number of nanoseconds for the current second
  auto durNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   tpNs.time_since_epoch()) %
               1000000000;
  // convert to std::time_t in order to convert to std::tm (broken time)
  auto tpTime = std::chrono::system_clock::to_time_t(tp);
  // convert to broken time
  os << std::put_time(std::localtime(&tpTime), "%d %b %Y %H:%M:%S") << '.'
     << std::setfill('0') << std::setw(9) << durNs.count();
  return os;
}

void BasicTimer::addTimerCallback(std::shared_ptr<TimerCallback> cb) {
  m_mutex.lock();
  auto itr = std::find(m_timerCallbacks.begin(), m_timerCallbacks.end(), cb);
  if (itr != m_timerCallbacks.end()) m_timerCallbacks.erase(itr);
  m_timerCallbacks.push_back(cb);
  std::sort(
      m_timerCallbacks.begin(), m_timerCallbacks.end(),
      [](std::shared_ptr<TimerCallback> a, std::shared_ptr<TimerCallback> b) {
        return a->getTimeout() < b->getTimeout();
      });
  m_mutex.unlock();
}

void BasicTimer::tick() {
  m_mutex.lock();
  m_ticks++;
  std::chrono::duration<double> ct_d(m_timeStep);
  m_simTime += std::chrono::duration_cast<std::chrono::nanoseconds>(ct_d);
  m_mutex.unlock();

  // Check if any timers should trigger
  int numCBs = m_timerCallbacks.size();
  for (int i = 0; i < numCBs; i++) {
    m_mutex.lock();
    bool isEmpty = m_timerCallbacks.empty();
    m_mutex.unlock();
    if (isEmpty) break;

    m_mutex.lock();
    std::shared_ptr<TimerCallback> cb = m_timerCallbacks.back();
    bool execTimer = cb->getTimeout() <= m_simTime && !cb->isPaused();
    m_mutex.unlock();
    if (!execTimer) break;

    m_mutex.lock();
    m_timerCallbacks.pop_back();
    m_mutex.unlock();
    cb->run(m_ticks, m_simTime);

    if (cb->isRepeating()) {
      std::chrono::duration<double> overshoot =
          (cb->getTimeout() < m_simTime) ? std::chrono::duration<double>(0)
                                         : m_simTime - cb->getTimeout();
      sim_clock_t::time_point nextTimeout =
          m_simTime +
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              cb->getRepeatTime()) -
          std::chrono::duration_cast<std::chrono::nanoseconds>(overshoot);

      cb->setTimeout(nextTimeout);
      std::cout << "setting to " << nextTimeout << std::endl;
      addTimerCallback(cb);
    }
  }
}
