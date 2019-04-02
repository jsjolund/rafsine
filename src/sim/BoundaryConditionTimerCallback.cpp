#include "BoundaryConditionTimerCallback.hpp"

BoundaryConditionTimerCallback::BoundaryConditionTimerCallback(
    KernelInterface* kernel, std::vector<BoundaryCondition>* bcs,
    std::shared_ptr<VoxelGeometry> voxelGeometry,
    std::shared_ptr<UnitConverter> uc, std::string inputCsvPath)
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
    m_numRows = m_csv.GetColumn<uint64_t>("time").size();
    std::cout << "Input CSV contains " << m_numRows << " rows" << std::endl;
  }
}

void BoundaryConditionTimerCallback::run(uint64_t ticks) {
  if (m_inputCsvPath.length() == 0) return;
  if (m_rowIdx >= m_numRows) return;

  std::cout << "Setting boundary conditions (row " << m_rowIdx << " of "
            << m_numRows << ")" << std::endl;

  std::vector<std::string> headers = m_csv.GetColumnNames();
  for (int col = 0; col < headers.size(); col++) {
    const std::string header = headers.at(col);

    if (header.length() > 2) {
      std::string name =
          std::string(header).erase(header.length() - 2, header.length() - 1);
      if (endsWithCaseInsensitive(header, "_T")) {
        real temp = m_csv.GetCell<real>(col, m_rowIdx);
        std::cout << name << " temp=" << temp << std::endl;
      } else if (endsWithCaseInsensitive(header, "_Q")) {
        real flow = m_csv.GetCell<real>(col, m_rowIdx);
        std::cout << name << " flow=" << flow << std::endl;
      }
    }
  }

  int64_t t0 = m_csv.GetCell<uint64_t>(0, m_rowIdx);
  int64_t t1 = m_csv.GetCell<uint64_t>(0, m_rowIdx + 1);
  int64_t dt = t1 - t0;
  timeval repeatTime{.tv_sec = dt, .tv_usec = 0};
  setRepeatTime(repeatTime);
  m_rowIdx++;
}
