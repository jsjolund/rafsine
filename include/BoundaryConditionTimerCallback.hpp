#pragma once

#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <sys/time.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rapidcsv.h"

#include "BasicTimer.hpp"
#include "KernelInterface.hpp"
#include "UnitConverter.hpp"

class BoundaryConditionTimerCallback : public TimerCallback {
 private:
  std::shared_ptr<KernelInterface> m_kernel;
  std::shared_ptr<UnitConverter> m_uc;
  std::shared_ptr<BoundaryConditions> m_bcs;
  std::shared_ptr<VoxelGeometry> m_voxelGeometry;
  std::string m_inputCsvPath;
  unsigned int m_rowIdx;
  unsigned int m_numRows;
  rapidcsv::Document m_csv;

 public:
  BoundaryConditionTimerCallback& operator=(
      const BoundaryConditionTimerCallback& other) {
    TimerCallback::operator=(other);
    m_uc = other.m_uc;
    m_kernel = other.m_kernel;
    m_inputCsvPath = other.m_inputCsvPath;
    m_rowIdx = other.m_rowIdx;
    m_numRows = other.m_numRows;
    m_csv =
        rapidcsv::Document(other.m_inputCsvPath, rapidcsv::LabelParams(0, -1));
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
  static std::time_t parseCsvDatetime(std::string datetime) {
    std::tm tm = {};
    std::stringstream ss(datetime);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    return std::mktime(&tm);
  }
  void run(uint64_t simTicks, sim_clock_t::time_point simTime);
  void reset();
};
