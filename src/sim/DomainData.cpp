#include "DomainData.hpp"

void DomainData::loadSimulation(int nd,
                                std::string buildGeometryPath,
                                std::string settingsPath) {
  LuaData::loadSimulation(buildGeometryPath, settingsPath);

  m_bcs = m_voxGeo->getBoundaryConditions();
  m_avgs = m_voxGeo->getSensors();
  if (m_avgPeriod <= 0.0) {
    std::cout << "Invalid averaging period set " << m_avgPeriod << std::endl;
    m_avgs->clear();
  }
  std::cout << "Number of lattice site types: " << m_voxGeo->getNumTypes()
            << std::endl;

  std::cout << "Allocating GPU resources" << std::endl;
  std::shared_ptr<VoxelArray> voxArray = m_voxGeo->getVoxelArray();
  voxArray->upload();
  m_kernel = std::make_shared<KernelInterface>(
      m_nx, m_ny, m_nz, m_unitConverter->N_to_s(1), m_param, m_bcs, voxArray,
      m_avgs, nd, m_partitioning);
  voxArray->deallocate(MemoryType::DEVICE_MEMORY);

  m_timer = std::make_shared<SimulationTimer>(m_nx * m_ny * m_nz,
                                              m_unitConverter->N_to_s(1));
}
