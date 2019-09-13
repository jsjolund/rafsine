#pragma once

#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <sys/time.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "rapidcsv.h"

#include "KernelInterface.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"

class BoundaryConditionTimerCallback : public SimulationTimerCallback {
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

  void run(uint64_t simTicks, timeval simTime);
};
