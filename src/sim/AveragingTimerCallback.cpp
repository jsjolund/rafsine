#include "AveragingTimerCallback.hpp"

AveragingTimerCallback::AveragingTimerCallback(
    std::shared_ptr<KernelInterface> kernel, std::shared_ptr<UnitConverter> uc,
    std::vector<VoxelVolume> avgVols, std::string outputCSVPath)
    : SimulationTimerCallback(),
      m_outputCsvPath(QString::fromStdString(outputCSVPath)),
      m_kernel(kernel),
      m_lastTicks(0),
      m_avgVols(avgVols),
      m_avgs(avgVols.size()),
      m_uc(uc) {
  if (m_outputCsvPath.length() > 0) {
    QFile outputCsv(m_outputCsvPath);
    // Overwrite any old output CSV
    QFileInfo outputCsvInfo(outputCsv);
    if (outputCsvInfo.size() > 0) {
      outputCsv.remove();
    }
    if (outputCsv.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream stream(&outputCsv);
      writeAveragesHeaders(stream);
    } else {
      throw std::runtime_error("Failed to open output CSV file");
    }
  } else {
    QTextStream stream(stdout, QIODevice::WriteOnly);
    writeAveragesHeaders(stream);
  }
}

void AveragingTimerCallback::writeAveragesHeaders(QTextStream &stream) {
  stream << "time,";
  for (int i = 0; i < m_avgs.size(); i++) {
    VoxelVolume avgVol = m_avgVols.at(i);
    QString name = QString::fromStdString(avgVol.getName());
    stream << name << "_T," << name << "_Q";
    if (i == m_avgs.size() - 1)
      stream << endl;
    else
      stream << ",";
  }
}

void AveragingTimerCallback::writeAverages(QTextStream &stream, uint64_t ticks,
                                           uint64_t avgTicks) {
  stream << ticks << ",";

  for (int i = 0; i < m_avgs.size(); i++) {
    VoxelVolume avgVol = m_avgVols.at(i);
    Average avg = m_kernel->getAverage(avgVol, avgTicks);
    // Temperature
    real temperature = m_uc->luTemp_to_Temp(avg.m_temperature);
    // Velocity magnitude
    real velocity = m_uc->C_U() * sqrt(avg.m_velocityX * avg.m_velocityX +
                                       avg.m_velocityY * avg.m_velocityY +
                                       avg.m_velocityZ * avg.m_velocityZ);
    // Flow through area
    real flow =
        velocity * avgVol.getNumVoxels() * pow(m_uc->C_L(), avgVol.getRank());

    stream << temperature << "," << flow;
    if (i == m_avgs.size() - 1)
      stream << endl;
    else
      stream << ",";
  }
  stream.flush();
}

void AveragingTimerCallback::run(uint64_t ticks) {
  if (m_avgVols.size() == 0) return;

  const uint64_t deltaTicks = ticks - m_lastTicks;
  m_lastTicks = ticks;
  // When timer callback is triggered, simulation has not been invoced for
  // this tick yet, and averages are read from last invocation, so subtract 2
  const uint64_t avgTicks = deltaTicks - 2;

  if (m_outputCsvPath.length() > 0) {
    QFile outputCsv(m_outputCsvPath);
    if (outputCsv.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream stream(&outputCsv);
      writeAverages(stream, ticks, avgTicks);
    } else {
      throw std::runtime_error("Failed to open output CSV file");
    }
  } else {
    QTextStream stream(stdout, QIODevice::WriteOnly);
    writeAverages(stream, ticks, avgTicks);
  }

  m_kernel->resetAverages();
}
