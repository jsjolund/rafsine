#pragma once

#include <sys/time.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "Average.hpp"
#include "AverageObserver.hpp"
#include "KernelInterface.hpp"
#include "Observable.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"

class AverageObservable : public Observable<AverageObserver> {
 public:
  void sendNotifications(const AverageMatrix& avgs) { notifyObservers(avgs); }
};

class AveragingTimerCallback : public SimulationTimerCallback,
                               public AverageObservable {
 private:
  std::shared_ptr<KernelInterface> m_kernel;
  std::shared_ptr<UnitConverter> m_uc;
  std::vector<VoxelVolume> m_avgVols;
  AverageMatrix m_matrix;
  uint64_t m_lastTicks;

 public:
  AveragingTimerCallback& operator=(const AveragingTimerCallback& other) {
    SimulationTimerCallback::operator=(other);
    // TODO(matrix?)
    m_kernel = other.m_kernel;
    m_uc = other.m_uc;
    m_avgVols = other.m_avgVols;
    m_lastTicks = other.m_lastTicks;
    return *this;
  }

  AveragingTimerCallback()
      : SimulationTimerCallback(),
        AverageObservable(),
        m_uc(NULL),
        m_kernel(NULL),
        m_lastTicks(0),
        m_matrix(),
        m_avgVols() {}

  AveragingTimerCallback(std::shared_ptr<KernelInterface> kernel,
                         std::shared_ptr<UnitConverter> uc,
                         std::vector<VoxelVolume> avgVols);

  void run(uint64_t simTicks, timeval simTime);
  void reset();
};
