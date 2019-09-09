#pragma once

#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "KernelInterface.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"

class AveragingTimerCallback : public SimulationTimerCallback {
 private:
  std::shared_ptr<KernelInterface> m_kernel;
  std::vector<VoxelVolume> m_avgAreas;
  std::vector<Average> m_avgs;
  QString m_outputCsvPath;
  std::shared_ptr<UnitConverter> m_uc;

 public:
  uint64_t m_lastTicks;

  AveragingTimerCallback& operator=(const AveragingTimerCallback& other) {
    m_uc = other.m_uc;
    m_kernel = other.m_kernel;
    m_avgAreas = other.m_avgAreas;
    m_avgs = other.m_avgs;
    m_outputCsvPath = other.m_outputCsvPath;
    m_lastTicks = other.m_lastTicks;
    return *this;
  }

  AveragingTimerCallback()
      : SimulationTimerCallback(),
        m_uc(NULL),
        m_kernel(NULL),
        m_lastTicks(0),
        m_avgAreas(),
        m_avgs(0) {}

  AveragingTimerCallback(std::shared_ptr<KernelInterface> kernel,
                         std::shared_ptr<UnitConverter> uc,
                         std::vector<VoxelVolume> avgAreas,
                         std::string outputCSVPath);

  void writeAverages(QTextStream& stream, uint64_t ticks, uint64_t avgTicks);
  void writeAveragesHeaders(QTextStream& stream);
  void run(uint64_t ticks);
};
