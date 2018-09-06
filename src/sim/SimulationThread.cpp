#include "SimulationThread.hpp"

SimulationThread::~SimulationThread()
{
  if (isRunning())
  {
    cancel();
    join();
  }
}

SimulationThread::SimulationThread()
    : OpenThreads::Thread(),
      m_paused(false),
      m_exit(false),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE),
      m_domainData(NULL)
{
}

SimulationThread::SimulationThread(DomainData *domainData)
    : OpenThreads::Thread(),
      m_domainData(domainData),
      m_paused(false),
      m_exit(false),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE)
{
  setDomainData(domainData);
}

DomainData *SimulationThread::getDomainData()
{
  return m_domainData;
}

void SimulationThread::setDomainData(DomainData *domainData)
{
  SIM_HIGH_PRIO_LOCK
  m_domainData = domainData;
  int plotSize = m_domainData->m_voxGeo->getNx() * m_domainData->m_voxGeo->getNy() * m_domainData->m_voxGeo->getNz();
  m_plot = thrust::device_vector<real>(plotSize);
  SIM_HIGH_PRIO_UNLOCK
}

bool SimulationThread::hasDomainData()
{
  return m_domainData != NULL;
}

int SimulationThread::cancel()
{
  SIM_HIGH_PRIO_LOCK
  m_exit = true;
  SIM_HIGH_PRIO_UNLOCK
  return OpenThreads::Thread::cancel();
}

// Upload new boundary conditions
void SimulationThread::uploadBCs()
{
  SIM_HIGH_PRIO_LOCK
  // k->uploadBCs();
  SIM_HIGH_PRIO_UNLOCK
}

// Reset the averaging array
void SimulationThread::resetAverages()
{
  if (!m_domainData)
    return;
  SIM_HIGH_PRIO_LOCK
  m_domainData->m_kernelData->resetAverages();
  SIM_HIGH_PRIO_UNLOCK
}

// Reset the simulation
void SimulationThread::resetDfs()
{
  if (!m_domainData)
    return;
  SIM_HIGH_PRIO_LOCK
  m_domainData->m_simTimer->reset();
  m_domainData->m_kernelData->resetAverages();
  m_domainData->m_kernelData->initDomain(1.0, 0, 0, 0, m_domainData->m_kernelParam->Tinit);
  SIM_HIGH_PRIO_UNLOCK
}

void SimulationThread::pause(bool state)
{
  SIM_HIGH_PRIO_LOCK
  m_paused = state;
  SIM_HIGH_PRIO_UNLOCK
}

bool SimulationThread::isPaused()
{
  return m_paused;
}

// Redraw the visualization plot
void SimulationThread::draw(real *plot, DisplayQuantity::Enum visQ)
{
  SIM_HIGH_PRIO_LOCK
  if (visQ != m_visQ)
  {
    m_visQ = visQ;
    m_domainData->m_kernelData->compute(thrust::raw_pointer_cast(&(m_plot)[0]), m_visQ);
    m_domainData->m_simTimer->tick();
  }
  int plotSize = m_domainData->m_voxGeo->getNx() * m_domainData->m_voxGeo->getNy() * m_domainData->m_voxGeo->getNz();
  thrust::device_ptr<real> dp1(thrust::raw_pointer_cast(&(m_plot)[0]));
  thrust::device_ptr<real> dp2(plot);
  thrust::copy(dp1, dp1 + plotSize, dp2);
  SIM_HIGH_PRIO_UNLOCK

  cudaDeviceSynchronize();
}

void SimulationThread::run()
{
  while (true)
  {
    SIM_LOW_PRIO_LOCK
    if (!m_paused)
    {
      m_domainData->m_kernelData->compute(thrust::raw_pointer_cast(&(m_plot)[0]), m_visQ);
      m_domainData->m_simTimer->tick();
    }
    SIM_LOW_PRIO_UNLOCK

    if (m_exit)
      return;

    cudaDeviceSynchronize();
  }
}