#include "SimulationWorker.hpp"

SimulationWorker::~SimulationWorker()
{
  delete m_domainData;
  cudaStreamDestroy(m_simStream);
}

SimulationWorker::SimulationWorker()
    : m_domainData(NULL),
      m_exit(false),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE),
      m_simStream(0)
{
}

SimulationWorker::SimulationWorker(DomainData *domainData)
    : m_domainData(domainData),
      m_exit(false),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE),
      m_simStream(0)
{
  setDomainData(domainData);
}

void SimulationWorker::setDomainData(DomainData *domainData)
{
  SIM_HIGH_PRIO_LOCK
  if (m_domainData)
    delete m_domainData;
  m_domainData = domainData;
  int plotSize = m_domainData->m_voxGeo->getNx() * m_domainData->m_voxGeo->getNy() * m_domainData->m_voxGeo->getNz();
  m_plot = thrust::device_vector<real>(plotSize);
  SIM_HIGH_PRIO_UNLOCK
}

bool SimulationWorker::hasDomainData()
{
  return m_domainData != NULL;
}

int SimulationWorker::cancel()
{
  SIM_HIGH_PRIO_LOCK
  m_exit = true;
  SIM_HIGH_PRIO_UNLOCK
  return 0;
}

int SimulationWorker::resume()
{
  SIM_HIGH_PRIO_LOCK
  m_exit = false;
  SIM_HIGH_PRIO_UNLOCK
  return 0;
}

// Upload new boundary conditions
void SimulationWorker::uploadBCs()
{
  SIM_HIGH_PRIO_LOCK
  m_domainData->m_kernelData->uploadBCs(m_domainData->m_bcs);
  SIM_HIGH_PRIO_UNLOCK
}

// Reset the averaging array
void SimulationWorker::resetAverages()
{
  if (!m_domainData)
    return;
  SIM_HIGH_PRIO_LOCK
  m_domainData->m_kernelData->resetAverages();
  SIM_HIGH_PRIO_UNLOCK
}

// Reset the simulation
void SimulationWorker::resetDfs()
{
  if (!m_domainData)
    return;
  SIM_HIGH_PRIO_LOCK
  m_domainData->m_simTimer->reset();
  m_domainData->m_kernelData->resetAverages();
  m_domainData->m_kernelData->initDomain(1.0, 0, 0, 0,
                                         m_domainData->m_kernelParam->Tinit);
  SIM_HIGH_PRIO_UNLOCK
}

// Redraw the visualization plot
void SimulationWorker::draw(real *plot, DisplayQuantity::Enum visQ)
{
  SIM_HIGH_PRIO_LOCK
  if (visQ != m_visQ)
  {
    m_visQ = visQ;
    m_domainData->m_kernelData->compute(thrust::raw_pointer_cast(&(m_plot)[0]),
                                        m_visQ,
                                        m_simStream);
    m_domainData->m_simTimer->tick();
  }
  thrust::device_ptr<real> dp1(thrust::raw_pointer_cast(&(m_plot)[0]));
  thrust::device_ptr<real> dp2(plot);
  thrust::copy(dp1, dp1 + m_plot.size(), dp2);
  SIM_HIGH_PRIO_UNLOCK
}

void SimulationWorker::run()
{
  int priorityHigh, priorityLow;
  cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
  cudaStreamCreateWithPriority(&m_simStream, cudaStreamNonBlocking, priorityHigh);

  while (!m_exit)
  {
    SIM_LOW_PRIO_LOCK
    m_domainData->m_kernelData->compute(thrust::raw_pointer_cast(&(m_plot)[0]),
                                        m_visQ,
                                        m_simStream);
    m_domainData->m_simTimer->tick();
    SIM_LOW_PRIO_UNLOCK
    cudaDeviceSynchronize();
  }
  emit finished();
}