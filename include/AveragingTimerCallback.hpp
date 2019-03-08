#pragma once

#include <vector>

#include "KernelInterface.hpp"
#include "SimulationTimer.hpp"

class AveragingTimerCallback : public SimulationTimerCallback {
 private:
  real m_C_U;
  real m_C_L;
  KernelInterface *m_kernel;
  std::vector<VoxelArea> m_avgAreas;
  std::vector<Average> m_avgs;

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

  AveragingTimerCallback(KernelInterface *kernel,
                         std::vector<VoxelArea> avgAreas, real C_U, real C_L)
      : SimulationTimerCallback(),
        m_kernel(kernel),
        m_C_U(C_U),
        m_C_L(C_L),
        m_lastTicks(0),
        m_avgAreas(avgAreas),
        m_avgs(avgAreas.size()) {}

  void run(uint64_t ticks) {
    // if (m_avgAreas.size() == 0) return;
    // const uint64_t deltaTicks = ticks - m_lastTicks;
    // m_lastTicks = ticks;

    // for (int i = 0; i < m_avgs.size(); i++) {
    //   VoxelArea avgArea = m_avgAreas.at(i);
    //   Average avg = m_kernel->getAverage(avgArea, deltaTicks - 1);
    //   std::cout << avgArea.m_name << " temp=" << avg.m_temperature << " ";
    // }
    // std::cout << std::endl;
    // m_kernel->resetAverages();
  }
};
