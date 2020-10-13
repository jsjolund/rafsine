#pragma once

#include <sys/time.h>

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "BasicTimer.hpp"
#include "KernelInterface.hpp"
#include "UnitConverter.hpp"
#include "rapidcsv.h"

class BoundaryConditionTimerCallback : public TimerCallback {
 private:
  std::shared_ptr<KernelInterface> m_kernel;
  std::shared_ptr<UnitConverter> m_uc;
  std::shared_ptr<BoundaryConditions> m_bcs;
  std::shared_ptr<VoxelGeometry> m_voxelGeometry;
  std::string m_inputCsvPath;
  unsigned int m_rowIdx;

 public:
  BoundaryConditionTimerCallback& operator=(
      const BoundaryConditionTimerCallback& other) {
    TimerCallback::operator=(other);
    m_uc = other.m_uc;
    m_kernel = other.m_kernel;
    m_inputCsvPath = other.m_inputCsvPath;
    m_rowIdx = other.m_rowIdx;
    return *this;
  }

  BoundaryConditionTimerCallback(std::shared_ptr<KernelInterface> kernel,
                                 std::shared_ptr<BoundaryConditions> bcs,
                                 std::shared_ptr<VoxelGeometry> voxelGeometry,
                                 std::shared_ptr<UnitConverter> uc,
                                 std::string inputCsvPath);

  bool endsWithCaseInsensitive(std::string mainStr, std::string toMatch) {
    auto it = toMatch.begin();
    return mainStr.size() >= toMatch.size() &&
           std::all_of(
               std::next(mainStr.begin(), mainStr.size() - toMatch.size()),
               mainStr.end(), [&it](const char& c) {
                 return ::tolower(c) == ::tolower(*(it++));
               });
  }

  void run(uint64_t simTicks, sim_clock_t::time_point simTime);
  void reset();
};
