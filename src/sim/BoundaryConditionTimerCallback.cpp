#include "BoundaryConditionTimerCallback.hpp"

BoundaryConditionTimerCallback::BoundaryConditionTimerCallback(
    std::shared_ptr<KernelInterface> kernel,
    std::shared_ptr<BoundaryConditions> bcs,
    std::shared_ptr<VoxelGeometry> voxelGeometry,
    std::shared_ptr<UnitConverter> uc, std::string inputCsvPath)
    : SimulationTimerCallback(),
      m_inputCsvPath(inputCsvPath),
      m_kernel(kernel),
      m_voxelGeometry(voxelGeometry),
      m_bcs(bcs),
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

void BoundaryConditionTimerCallback::run(uint64_t simTicks, timeval simTime) {
  if (m_inputCsvPath.length() == 0) return;
  if (m_rowIdx >= m_numRows) return;

  std::cout << "Setting boundary conditions (row " << m_rowIdx << " of "
            << m_numRows << ")";

  if (m_rowIdx < m_numRows - 1) {
    int64_t t0 = m_csv.GetCell<uint64_t>(0, m_rowIdx);
    int64_t t1 = m_csv.GetCell<uint64_t>(0, m_rowIdx + 1);
    int64_t dt = t1 - t0;
    timeval repeatTime{.tv_sec = dt, .tv_usec = 0};
    setRepeatTime(repeatTime);
    std::cout << " next update at " << timeval{.tv_sec = t1, .tv_usec = 0}
              << std::endl;
  } else {
    std::cout << std::endl;
  }

  std::vector<std::string> headers = m_csv.GetColumnNames();
  // Parse all columns except the first (time)
  for (int col = 1; col < headers.size(); col++) {
    const std::string header = headers.at(col);

    if (header.length() <= 2) continue;

    std::string name =
        std::string(header).erase(header.length() - 2, header.length() - 1);
    std::unordered_set<VoxelQuad> quads = m_voxelGeometry->getQuadsByName(name);

    if (endsWithCaseInsensitive(header, "_T")) {
      // If the header name ends with _T it is temperature
      real temp = m_csv.GetCell<real>(col, m_rowIdx);
      // std::cout << name << " temp=" << temp << std::endl;

      for (VoxelQuad quad : quads) {
        BoundaryCondition *bc = &(m_bcs->at(quad.m_bc.m_id));
        bc->m_temperature = m_uc->Temp_to_lu(temp);
      }

    } else if (endsWithCaseInsensitive(header, "_Q")) {
      // If the header name ends with _Q it is volumetric flow
      real flow = m_csv.GetCell<real>(col, m_rowIdx);
      // std::cout << name << " flow=" << flow << std::endl;

      for (VoxelQuad quad : quads) {
        BoundaryCondition *bc = &(m_bcs->at(quad.m_bc.m_id));
        real velocityLu = max(0.0, m_uc->Q_to_Ulu(flow, quad.getAreaReal()));
        glm::vec3 nVelocity = glm::normalize(
            glm::vec3(bc->m_normal.x, bc->m_normal.y, bc->m_normal.z));
        if (bc->m_type == VoxelType::INLET_ZERO_GRADIENT)
          nVelocity = -nVelocity;
        bc->m_velocity.x = nVelocity.x * velocityLu;
        bc->m_velocity.y = nVelocity.y * velocityLu;
        bc->m_velocity.z = nVelocity.z * velocityLu;
      }
    }
  }
  m_kernel->uploadBCs(m_bcs);

  m_rowIdx++;
}
