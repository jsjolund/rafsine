#pragma once

#include <sys/time.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "Average.hpp"
#include "AverageObserver.hpp"
#include "BasicTimer.hpp"
#include "KernelInterface.hpp"
#include "Observable.hpp"
#include "UnitConverter.hpp"

/**
 * @brief Base class for averaging observable enitites
 *
 */
class AverageObservable : public Observable<AverageObserver> {
 public:
  /**
   * @brief Notifies observers of new averaging values
   *
   * @param avgs
   */
  void sendNotifications(const AverageMatrix& avgs) { notifyObservers(avgs); }
};

/**
 * @brief Simulation timer callback for new averaging values. Sends values to
 * observers at specified times.
 */
class AveragingTimerCallback : public TimerCallback, public AverageObservable {
 private:
  std::shared_ptr<KernelInterface> m_kernel;
  std::shared_ptr<UnitConverter> m_uc;
  std::vector<VoxelCuboid> m_avgVols;
  AverageMatrix m_matrix;
  uint64_t m_lastTicks;

 public:
  AveragingTimerCallback& operator=(const AveragingTimerCallback& other) {
    TimerCallback::operator=(other);
    // TODO(matrix?)
    m_kernel = other.m_kernel;
    m_uc = other.m_uc;
    m_avgVols = other.m_avgVols;
    m_lastTicks = other.m_lastTicks;
    return *this;
  }

  AveragingTimerCallback()
      : TimerCallback(),
        AverageObservable(),
        m_kernel(NULL),
        m_uc(NULL),
        m_avgVols(),
        m_matrix(),
        m_lastTicks(0) {}

  AveragingTimerCallback(std::shared_ptr<KernelInterface> kernel,
                         std::shared_ptr<UnitConverter> uc,
                         std::vector<VoxelCuboid> avgVols);

  void run(uint64_t simTicks, sim_clock_t::time_point simTime);
  void reset();
};
