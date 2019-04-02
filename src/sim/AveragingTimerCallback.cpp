#include "AveragingTimerCallback.hpp"

AveragingTimerCallback::AveragingTimerCallback(
    KernelInterface* kernel, std::shared_ptr<UnitConverter> uc,
    std::vector<VoxelArea> avgAreas, std::string outputCSVPath)
    : SimulationTimerCallback(),
      m_outputCsvPath(QString::fromStdString(outputCSVPath)),
      m_kernel(kernel),
      m_lastTicks(0),
      m_avgAreas(avgAreas),
      m_avgs(avgAreas.size()),
      m_uc(uc) {
  if (m_outputCsvPath.length() > 0) {
    QFile outputCsv(m_outputCsvPath);
    QFileInfo outputCsvInfo(outputCsv);
    if (outputCsvInfo.size() > 0) {
      outputCsv.remove();
    }
    if (outputCsv.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream stream(&outputCsv);
      stream << "time,";
      for (int i = 0; i < m_avgs.size(); i++) {
        VoxelArea avgArea = m_avgAreas.at(i);
        QString name = QString::fromStdString(avgArea.getName());
        stream << name << "_T," << name << "_Q";
        if (i == m_avgs.size() - 1)
          stream << endl;
        else
          stream << ",";
      }
    } else {
      throw std::runtime_error("Failed to open output CSV file");
    }
  }
}

void AveragingTimerCallback::run(uint64_t ticks) {
  if (m_avgAreas.size() == 0 || m_outputCsvPath.length() == 0) return;

  const uint64_t deltaTicks = ticks - m_lastTicks;
  m_lastTicks = ticks;
  // When timer callback is triggered, simulation has not been invoced for
  // this tick yet, and averages are read from last invocation, so subtract 2
  const uint64_t avgTicks = deltaTicks - 2;
  QFile outputCsv(m_outputCsvPath);
  if (outputCsv.open(QIODevice::WriteOnly | QIODevice::Append)) {
    QTextStream stream(&outputCsv);
    stream << ticks << ",";
    for (int i = 0; i < m_avgs.size(); i++) {
      VoxelArea avgArea = m_avgAreas.at(i);
      Average avg = m_kernel->getAverage(avgArea, avgTicks);
      // Temperature
      real temperature = m_uc->luTemp_to_Temp(avg.m_temperature);
      // Velocity magnitude
      real velocity = m_uc->C_U() * sqrt(avg.m_velocityX * avg.m_velocityX +
                                         avg.m_velocityY * avg.m_velocityY +
                                         avg.m_velocityZ * avg.m_velocityZ);
      // Flow through area
      real flow = velocity * avgArea.getNumVoxels() *
                  pow(m_uc->C_L(), avgArea.getRank());

      stream << temperature << "," << flow;
      if (i == m_avgs.size() - 1)
        stream << endl;
      else
        stream << ",";
    }
  } else {
    throw std::runtime_error("Failed to open output CSV file");
  }
  m_kernel->resetAverages();
}
