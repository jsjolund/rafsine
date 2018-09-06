#include "SimulationTimer.hpp"

int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec)
  {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000)
  {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }
  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  if (result)
  {
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;
  }
  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

void timeval_add(timeval *t, double seconds)
{
  int secStep = (int)seconds;
  int usecStep = (int)((seconds - secStep) * 1e6);
  t->tv_sec += secStep;
  t->tv_usec += usecStep;
  if (t->tv_usec > 1000000)
  {
    t->tv_sec += t->tv_usec / 1000000;
    t->tv_usec = t->tv_usec % 1000000;
  }
}

std::ostream &operator<<(std::ostream &os, const SimulationTimer &timer)
{
  const timeval simTime = timer.getSimulationTime();
  struct tm *nowtm;
  char tmbuf[64];
  nowtm = gmtime(&simTime.tv_sec);
  strftime(tmbuf, sizeof tmbuf, "%d-%b-%Y %H:%M:%S", nowtm);
  return os << tmbuf;
}

SimulationTimer::SimulationTimer(unsigned int latticeSize, double secSimPerUpdate)
    : m_latticeSize(latticeSize),
      m_secSimPerUpdate(secSimPerUpdate),
      m_latticeUpdateCounter(0),
      m_currentLups(0),
      m_currentMlups(0)
{
  m_statsTimer.setStartTick();
  m_simTime.tv_sec = 0;
  m_simTime.tv_usec = 0;
}

void SimulationTimer::setSimulationTime(timeval newTime)
{
  m_mutex.lock();
  m_simTime = newTime;
  m_mutex.unlock();
}

void SimulationTimer::reset()
{
  m_latticeUpdateCounter = 0,
  m_currentLups = 0,
  m_currentMlups = 0;
  m_simTime.tv_sec = 0;
  m_simTime.tv_usec = 0;
  m_statsTimer.setStartTick();
}

void SimulationTimer::addSimulationTimeout(SimulationTimerCallback *cb)
{
  m_mutex.lock();
  m_timerCallbacks.push_back(cb);
  std::sort(m_timerCallbacks.begin(),
            m_timerCallbacks.end(),
            [](SimulationTimerCallback *a, SimulationTimerCallback *b) {
              timeval valA = a->m_timeout;
              timeval valB = b->m_timeout;
              return timeval_subtract(NULL, &valB, &valA);
            });
  m_mutex.unlock();
}

void SimulationTimer::tick()
{
  m_mutex.lock();
  m_latticeUpdateCounter++;
  timeval_add(&m_simTime, m_secSimPerUpdate);

  double statsTimeDelta = m_statsTimer.time_s();
  if (statsTimeDelta >= SIM_STATS_UPDATE_PERIOD)
  {
    m_currentMlups = (int)(1e-6 * m_latticeUpdateCounter * m_latticeSize / statsTimeDelta);
    m_currentLups = (int)(m_latticeUpdateCounter / statsTimeDelta);
    m_realTimeRate = m_latticeUpdateCounter * m_secSimPerUpdate / statsTimeDelta;
    m_latticeUpdateCounter = 0;
    m_statsTimer.setStartTick();
  }
  m_mutex.unlock();

  while (1)
  {
    m_mutex.lock();
    bool isEmpty = m_timerCallbacks.empty();
    m_mutex.unlock();
    if (isEmpty)
      break;
    m_mutex.lock();
    timeval cbTimeout = m_timerCallbacks.back()->m_timeout;
    timeval nowTime = m_simTime;
    int hasTimeout = timeval_subtract(NULL, &cbTimeout, &nowTime);
    m_mutex.unlock();
    if (!hasTimeout)
      break;
    m_mutex.lock();
    SimulationTimerCallback *handle = m_timerCallbacks.back();
    m_timerCallbacks.pop_back();
    m_mutex.unlock();
    handle->run();
    delete handle;
  }
}