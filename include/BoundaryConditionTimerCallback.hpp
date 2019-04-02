#pragma once

#include <QFile>
#include <QFileInfo>
#include <QTextStream>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rapidcsv.h"

#include "KernelInterface.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"

class BoundaryConditionTimerCallback : public SimulationTimerCallback {
 private:
  std::shared_ptr<KernelInterface> m_kernel;
  std::string m_inputCsvPath;
  std::shared_ptr<UnitConverter> m_uc;
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

  BoundaryConditionTimerCallback()
      : SimulationTimerCallback(),
        m_uc(NULL),
        m_kernel(NULL),
        m_rowIdx(0),
        m_numRows(0) {}

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

  void run(uint64_t ticks);
};
