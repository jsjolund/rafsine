#include "AveragingTimerCallback.hpp"

AveragingTimerCallback::AveragingTimerCallback(
    std::shared_ptr<KernelInterface> kernel,
    std::shared_ptr<UnitConverter> uc,
    std::vector<VoxelVolume> avgVols)
    : SimulationTimerCallback(),
      m_kernel(kernel),
      m_lastTicks(0),
      m_matrix(),
      m_avgVols(avgVols),
      m_uc(uc) {
  for (VoxelVolume vol : avgVols) m_matrix.m_columns.push_back(vol.getName());
}

void AveragingTimerCallback::reset() { m_lastTicks = 0; }

void AveragingTimerCallback::run(uint64_t ticks, timeval simTime) {
  if (m_avgVols.size() == 0) return;

  // When timer callback is triggered, simulation has not been invoked for
  // this tick yet, and averages are read from last invocation, so subtract 2
  const uint64_t deltaTicks = ticks - m_lastTicks;
  m_lastTicks = ticks;
  const uint64_t avgTicks = deltaTicks - 2;

  AverageData row;
  timevalToTimepoint(simTime, &row.m_time);  // TODO(subtract 2 steps?)

  for (int i = 0; i < m_avgVols.size(); i++) {
    VoxelVolume avgVol = m_avgVols.at(i);
    LatticeAverage lAvg = m_kernel->getAverage(avgVol, avgTicks);

    Average avg;
    avg.m_temperature = lAvg.getTemperature(*m_uc);
    avg.m_velocity = lAvg.getVelocity(*m_uc);
    avg.m_flow = lAvg.getFlow(*m_uc, avgVol);

    row.m_measurements.push_back(avg);
  }
  m_matrix.m_rows.push_back(row);

  sendNotifications(m_matrix);

  m_kernel->resetAverages();
}
