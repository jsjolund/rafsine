#include "DomainData.hpp"

void DomainData::loadSimulation(int numDevices,
                                std::string buildGeometryPath,
                                std::string settingsPath) {
  LuaData::loadSimulation(buildGeometryPath, settingsPath);

  m_bcs = m_voxGeo->getBoundaryConditions();
  m_avgs = m_voxGeo->getSensors();
  if (m_avgPeriod <= 0.0) {
    std::cout << "Invalid sensor averaging period set " << m_avgPeriod
              << " removing sensors..." << std::endl;
    m_avgs->clear();
  }
  std::cout << "Number of lattice site types: " << m_voxGeo->getNumTypes()
            << std::endl;

  std::cout << "Allocating GPU resources" << std::endl;
  std::shared_ptr<VoxelArray> voxArray = m_voxGeo->getVoxelArray();
  voxArray->upload();
  m_kernel = std::make_shared<KernelInterface>(
      m_nx, m_ny, m_nz, m_unitConverter->N_to_s(1), m_param, m_bcs, voxArray,
      m_avgs, numDevices);
  voxArray->deallocate(MemoryType::DEVICE_MEMORY);

  m_timer = std::make_shared<SimulationTimer>(m_nx * m_ny * m_nz,
                                              m_unitConverter->N_to_s(1));
}
