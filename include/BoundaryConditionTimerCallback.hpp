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
  KernelInterface* m_kernel;
  std::string m_inputCsvPath;
  std::shared_ptr<UnitConverter> m_uc;
  unsigned int m_rowIdx;
  unsigned int m_numRows;
  rapidcsv::Document m_csv;

 public:
  BoundaryConditionTimerCallback()
      : SimulationTimerCallback(),
        m_uc(NULL),
        m_kernel(NULL),
        m_rowIdx(0),
        m_numRows(0) {}

  BoundaryConditionTimerCallback(KernelInterface* kernel,
                                 std::shared_ptr<UnitConverter> uc,
                                 std::string inputCsvPath)
      : SimulationTimerCallback(),
        m_inputCsvPath(inputCsvPath),
        m_kernel(kernel),
        m_rowIdx(0),
        m_uc(uc) {
    if (m_inputCsvPath.length() > 0) {
      QFile inputCsv(QString::fromStdString(inputCsvPath));
      QFileInfo inputCsvInfo(inputCsv);
      if (!inputCsvInfo.isReadable())
        throw std::runtime_error("Failed to open input CSV file");
      m_csv = rapidcsv::Document(inputCsvPath, rapidcsv::LabelParams(0, -1));
      m_numRows = m_csv.GetRowCount();
    }
  }

  void run(uint64_t ticks) {
    if (m_inputCsvPath.length() == 0) return;
    int64_t t0 = m_csv.GetCell<uint64_t>(0, m_rowIdx);
    int64_t t1 = m_csv.GetCell<uint64_t>(0, m_rowIdx + 1);
    int64_t dt = t1 - t0;
    timeval repeatTime{.tv_sec = dt, .tv_usec = 0};
    setRepeatTime(repeatTime);
    m_rowIdx++;
    std::cout << "Setting boundary conditions..." << std::endl;
  }

  BoundaryConditionTimerCallback& operator=(
      const BoundaryConditionTimerCallback& other) {
    m_uc = other.m_uc;
    m_kernel = other.m_kernel;
    m_inputCsvPath = other.m_inputCsvPath;
    m_csv =
        rapidcsv::Document(other.m_inputCsvPath, rapidcsv::LabelParams(0, -1));
    return *this;
  }
};
