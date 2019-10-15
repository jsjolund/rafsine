#include "SimulationTimer.hpp"

int timeval_subtract(const struct timeval& x,
                     const struct timeval& y,
                     struct timeval* result) {
  timeval d = y;
  // Perform the carry for the later subtraction by updating y.
  if (x.tv_usec < y.tv_usec) {
    int nsec = (y.tv_usec - x.tv_usec) / 1000000 + 1;
    d.tv_usec -= 1000000 * nsec;
    d.tv_sec += nsec;
  }
  if (x.tv_usec - y.tv_usec > 1000000) {
    int nsec = (x.tv_usec - y.tv_usec) / 1000000;
    d.tv_usec += 1000000 * nsec;
    d.tv_sec -= nsec;
  }
  // Compute the time remaining to wait. tv_usec is certainly positive.
  if (result) {
    result->tv_sec = x.tv_sec - d.tv_sec;
    result->tv_usec = x.tv_usec - d.tv_usec;
  }
  // Return 1 if result is negative or zero
  return x.tv_sec <= d.tv_sec;
}

void timeval_add(const timeval& a, const timeval& b, timeval* result) {
  result->tv_sec = a.tv_sec + b.tv_sec;
  result->tv_usec = a.tv_usec + b.tv_usec;
  if (result->tv_usec >= 1000000) {
    result->tv_sec++;
    result->tv_usec -= 1000000;
  }
}

void timeval_add_seconds(const timeval& t,
                         const double seconds,
                         timeval* result) {
  timeval dt;
  dt.tv_sec = static_cast<int>(seconds);
  dt.tv_usec = static_cast<int>((seconds - dt.tv_sec) * 1e6);
  timeval_add(t, dt, result);
}

std::ostream& operator<<(std::ostream& os, const timeval& tval) {
  struct tm nowtm;
  char tmbuf[64];
  gmtime_r(&tval.tv_sec, &nowtm);
  strftime(tmbuf, sizeof tmbuf, "%d-%b-%Y %H:%M:%S", &nowtm);
  return os << tmbuf;
}

std::ostream& operator<<(std::ostream& os, const SimulationTimer& timer) {
  const timeval simTime = timer.getTime();
  struct tm nowtm;
  char tmbuf[64];
  gmtime_r(&simTime.tv_sec, &nowtm);
  strftime(tmbuf, sizeof tmbuf, "%d-%b-%Y %H:%M:%S", &nowtm);
  return os << tmbuf;
}

SimulationTimer::SimulationTimer(unsigned int latticeSize,
                                 double secSimPerUpdate)
    : m_latticeSize(latticeSize),
      m_secSimPerUpdate(secSimPerUpdate),
      m_simTime({.tv_sec = 0, .tv_usec = 0}),
      m_latticeUpdateCounter(0),
      m_currentLups(0),
      m_currentMlups(0),
      m_totalMlups(0),
      m_totalMlupsUpdates(0),
      m_ticks(0) {
  m_statsTimer.setStartTick();
}

void SimulationTimer::setTime(timeval newTime) {
  m_mutex.lock();
  m_simTime = newTime;
  m_mutex.unlock();
}

void SimulationTimer::setTime(int64_t newTime) {
  m_mutex.lock();
  m_simTime.tv_sec = newTime;
  m_mutex.unlock();
}

void SimulationTimer::reset() {
  m_timerCallbacks.clear();
  m_ticks = 0;
  m_latticeUpdateCounter = 0;
  m_currentLups = 0;
  m_currentMlups = 0;
  m_simTime.tv_sec = 0;
  m_simTime.tv_usec = 0;
  m_statsTimer.setStartTick();
}

void SimulationTimer::addSimulationTimer(
    std::shared_ptr<SimulationTimerCallback> cb) {
  m_mutex.lock();
  m_timerCallbacks.push_back(cb);
  std::sort(m_timerCallbacks.begin(), m_timerCallbacks.end(),
            [](std::shared_ptr<SimulationTimerCallback> a,
               std::shared_ptr<SimulationTimerCallback> b) {
              timeval valA = a->getTimeout();
              timeval valB = b->getTimeout();
              return timeval_subtract(valB, valA);
            });
  m_mutex.unlock();
}

void SimulationTimer::tick() {
  m_mutex.lock();
  m_ticks++;
  m_latticeUpdateCounter++;
  timeval_add_seconds(m_simTime, m_secSimPerUpdate, &m_simTime);

  double statsTimeDelta = m_statsTimer.time_s();
  if (statsTimeDelta >= SIM_STATS_UPDATE_PERIOD) {
    m_currentMlups = static_cast<int>(1e-6 * m_latticeUpdateCounter *
                                      m_latticeSize / statsTimeDelta);
    m_currentLups = static_cast<int>(m_latticeUpdateCounter / statsTimeDelta);
    m_totalMlups += m_currentMlups;
    m_totalMlupsUpdates++;

    m_realTimeRate =
        m_latticeUpdateCounter * m_secSimPerUpdate / statsTimeDelta;
    m_latticeUpdateCounter = 0;
    m_statsTimer.setStartTick();
  }
  m_mutex.unlock();

  // Check if any timers should trigger
  int numCBs = m_timerCallbacks.size();
  for (int i = 0; i < numCBs; i++) {
    m_mutex.lock();
    bool isEmpty = m_timerCallbacks.empty();
    m_mutex.unlock();
    if (isEmpty)
      break;

    m_mutex.lock();
    std::shared_ptr<SimulationTimerCallback> cb = m_timerCallbacks.back();
    int hasTimeout =
        timeval_subtract(cb->getTimeout(), m_simTime) && !cb->isPaused();
    m_mutex.unlock();
    if (!hasTimeout)
      break;

    m_mutex.lock();
    m_timerCallbacks.pop_back();
    m_mutex.unlock();
    cb->run(m_ticks, m_simTime);

    if (cb->isRepeating()) {
      timeval nextTimeout;
      timeval_add(m_simTime, cb->getRepeatTime(), &nextTimeout);
      cb->setTimeout(nextTimeout);
      addSimulationTimer(cb);
    }
  }
}
