#include "BoundaryConditionTimerCallback.hpp"

BoundaryConditionTimerCallback::BoundaryConditionTimerCallback(
    std::shared_ptr<KernelInterface> kernel,
    std::shared_ptr<BoundaryConditions> bcs,
    std::shared_ptr<VoxelGeometry> voxelGeometry,
    std::shared_ptr<UnitConverter> uc,
    std::string inputCsvPath)
    : TimerCallback(),
      m_kernel(kernel),
      m_uc(uc),
      m_bcs(bcs),
      m_voxelGeometry(voxelGeometry),
      m_inputCsvPath(inputCsvPath),
      m_rowIdx(0) {
  if (m_inputCsvPath.length() > 0) {
    QFile inputCsv(QString::fromStdString(inputCsvPath));
    QFileInfo inputCsvInfo(inputCsv);
    if (!inputCsvInfo.isReadable())
      throw std::runtime_error(ErrorFormat() << "Failed to open input CSV file "
                                             << m_inputCsvPath);
  }
}

void BoundaryConditionTimerCallback::reset() { m_rowIdx = 0; }

void BoundaryConditionTimerCallback::run(uint64_t, sim_clock_t::time_point) {
  if (m_inputCsvPath.length() == 0) {
    // No input csv provided
    return;
  }
  rapidcsv::Document csv(m_inputCsvPath, rapidcsv::LabelParams(0, -1));
  size_t numRows = csv.GetColumn<std::string>("time").size();
  if (m_rowIdx >= numRows) {
    // Finished reading all csv rows
    return;

  } else if (m_rowIdx == numRows - 1) {
    // Last row
    pause(true);

  } else {
    std::time_t t0 =
        BasicTimer::parseDatetime(csv.GetCell<std::string>(0, m_rowIdx));
    std::time_t t1 =
        BasicTimer::parseDatetime(csv.GetCell<std::string>(0, m_rowIdx + 1));
    sim_duration_t repeatTime =
        sim_clock_t::from_time_t(t1) - sim_clock_t::from_time_t(t0);
    setRepeatTime(repeatTime);
  }
  std::cout << "Setting boundary conditions (row " << m_rowIdx << " of "
            << numRows << ")" << std::endl;

  std::vector<std::string> headers = csv.GetColumnNames();
  // Parse all columns except the first (time)
  for (size_t col = 1; col < headers.size(); col++) {
    const std::string header = headers.at(col);

    if (header.length() <= 2) continue;

    // If the header name ends with _T it is temperature
    // If the header name ends with _Q it is volumetric flow
    std::string name =
        std::string(header).erase(header.length() - 2, header.length() - 1);
    std::unordered_set<VoxelQuad> quads = m_voxelGeometry->getQuadsByName(name);

    if (endsWithCaseInsensitive(header, "_T")) {
      real_t tempPhys = csv.GetCell<real_t>(col, m_rowIdx);
      for (VoxelQuad quad : quads) {
        BoundaryCondition* bc = &(m_bcs->at(quad.m_bc.m_id));
        bc->setTemperature(*m_uc, tempPhys);
      }

    } else if (endsWithCaseInsensitive(header, "_Q")) {
      real_t flowPhys = csv.GetCell<real_t>(col, m_rowIdx);
      for (VoxelQuad quad : quads) {
        BoundaryCondition* bc = &(m_bcs->at(quad.m_bc.m_id));
        bc->setFlow(*m_uc, flowPhys, quad.getAreaDiscrete(*m_uc));
      }
    }
  }
  m_kernel->uploadBCs(m_bcs);

  m_rowIdx++;
}
