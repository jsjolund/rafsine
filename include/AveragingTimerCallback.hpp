#pragma once

#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <cmath>
#include <string>
#include <vector>

#include "KernelInterface.hpp"
#include "SimulationTimer.hpp"

class AveragingTimerCallback : public SimulationTimerCallback {
 private:
  real m_C_U;
  real m_C_L;
  KernelInterface* m_kernel;
  std::vector<VoxelArea> m_avgAreas;
  std::vector<Average> m_avgs;
  QString m_outputCSVPath;

 public:
  uint64_t m_lastTicks;

  AveragingTimerCallback()
      : SimulationTimerCallback(),
        m_kernel(NULL),
        m_C_U(0),
        m_C_L(0),
        m_lastTicks(0),
        m_avgAreas(),
        m_avgs(0) {}

  AveragingTimerCallback(KernelInterface* kernel,
                         std::vector<VoxelArea> avgAreas, real C_U, real C_L,
                         std::string outputCSVPath)
      : SimulationTimerCallback(),
        m_outputCSVPath(QString::fromStdString(outputCSVPath)),
        m_kernel(kernel),
        m_C_U(C_U),
        m_C_L(C_L),
        m_lastTicks(0),
        m_avgAreas(avgAreas),
        m_avgs(avgAreas.size()) {
    QFile outputCSV(m_outputCSVPath);
    QFileInfo outputCSVinfo(outputCSV);
    if (outputCSVinfo.size() == 0 &&
        outputCSV.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream stream(&outputCSV);
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

  void run(uint64_t ticks) {
    if (m_avgAreas.size() == 0) return;
    const uint64_t deltaTicks = ticks - m_lastTicks;
    m_lastTicks = ticks;
    QFile outputCSV(m_outputCSVPath);
    if (outputCSV.open(QIODevice::WriteOnly | QIODevice::Append)) {
      QTextStream stream(&outputCSV);
      stream << ticks << ",";
      for (int i = 0; i < m_avgs.size(); i++) {
        VoxelArea avgArea = m_avgAreas.at(i);
        Average avg = m_kernel->getAverage(avgArea, deltaTicks - 1);
        // Temperature
        real temperature = avg.m_temperature;  // TODO(May need conversion)
        // Velocity magnitude
        real velocity = m_C_U * sqrt(avg.m_velocityX * avg.m_velocityX +
                                     avg.m_velocityY * avg.m_velocityY +
                                     avg.m_velocityZ * avg.m_velocityZ);
        // Flow through area
        real flow =
            velocity * avgArea.getNumVoxels() * pow(m_C_L, avgArea.getRank());

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

  AveragingTimerCallback& operator=(const AveragingTimerCallback& other) {
    m_C_U = other.m_C_U;
    m_C_L = other.m_C_L;
    m_kernel = other.m_kernel;
    m_avgAreas = other.m_avgAreas;
    m_avgs = other.m_avgs;
    m_outputCSVPath = other.m_outputCSVPath;
    return *this;
  }
};
