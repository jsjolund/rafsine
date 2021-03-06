#include "AveragingTimerCallback.hpp"

AveragingTimerCallback::AveragingTimerCallback(
    std::shared_ptr<KernelInterface> kernel,
    std::shared_ptr<UnitConverter> uc,
    std::vector<VoxelCuboid> avgVols)
    : TimerCallback(),
      m_kernel(kernel),
      m_uc(uc),
      m_avgVols(avgVols),
      m_matrix(),
      m_lastTicks(0) {
  for (VoxelCuboid vol : avgVols) m_matrix.m_columns.push_back(vol.getName());
}

void AveragingTimerCallback::reset() { m_lastTicks = 0; }

void AveragingTimerCallback::run(uint64_t ticks,
                                 sim_clock_t::time_point simTime) {
  if (m_avgVols.size() == 0) return;

  const uint64_t deltaTicks = ticks - m_lastTicks;
  m_lastTicks = ticks;

  AverageData row;
  row.m_time = simTime;

  m_kernel->calculateAverages();
  for (size_t i = 0; i < m_avgVols.size(); i++) {
    VoxelCuboid avgVol = m_avgVols.at(i);
    LatticeAverage lAvg = m_kernel->getAverage(avgVol, deltaTicks);

    Average avg;
    avg.temperature = lAvg.getTemperature(*m_uc);
    avg.velocity = lAvg.getVelocity(*m_uc);
    avg.flow = lAvg.getFlow(*m_uc, avgVol);

    row.m_measurements.push_back(avg);
  }
  m_matrix.m_rows.push_back(row);

  sendNotifications(m_matrix);

  m_kernel->resetAverages();
}
