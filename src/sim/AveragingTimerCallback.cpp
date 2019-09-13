#include "AveragingTimerCallback.hpp"

AveragingTimerCallback::AveragingTimerCallback(
    std::shared_ptr<KernelInterface> kernel, std::shared_ptr<UnitConverter> uc,
    std::vector<VoxelVolume> avgVols)
    : SimulationTimerCallback(),
      m_kernel(kernel),
      m_lastTicks(0),
      m_avgVols(avgVols),
      m_uc(uc) {}

void AveragingTimerCallback::run(uint64_t ticks, timeval simTime) {
  if (m_avgVols.size() == 0) return;

  const uint64_t deltaTicks = ticks - m_lastTicks;
  m_lastTicks = ticks;
  // When timer callback is triggered, simulation has not been invoced for
  // this tick yet, and averages are read from last invocation, so subtract 2
  const uint64_t avgTicks = deltaTicks - 2;

  AverageData data;
  data.time = simTime;
  for (int i = 0; i < m_avgVols.size(); i++) {
    VoxelVolume avgVol = m_avgVols.at(i);
    LatticeAverage lAvg = m_kernel->getAverage(avgVol, avgTicks);
    Average avg(*m_uc, avgVol, lAvg);
    data.rows.push_back(avg);
  }
  sendNotifications(data);

  m_kernel->resetAverages();
}
