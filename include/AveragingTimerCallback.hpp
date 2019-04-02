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
  KernelInterface* m_kernel;
  std::vector<VoxelArea> m_avgAreas;
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

  AveragingTimerCallback(KernelInterface* kernel,
                         std::shared_ptr<UnitConverter> uc,
                         std::vector<VoxelArea> avgAreas,
                         std::string outputCSVPath);

  void run(uint64_t ticks);
};
