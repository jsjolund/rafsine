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
      m_visQ(DisplayQuantity::Enum::TEMPERATURE)
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();

  m_domainData = new DomainData();
  int plotSize = m_domainData->m_voxGeo->getNx() * m_domainData->m_voxGeo->getNy() * m_domainData->m_voxGeo->getNz();
  m_plot = thrust::device_vector<real>(plotSize);

  m_m.unlock();
}

SimulationThread::SimulationThread(DomainData *domainData)
    : OpenThreads::Thread(),
      m_domainData(domainData),
      m_paused(false),
      m_exit(false),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE)
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();

  int plotSize = m_domainData->m_voxGeo->getNx() * m_domainData->m_voxGeo->getNy() * m_domainData->m_voxGeo->getNz();
  m_plot = thrust::device_vector<real>(plotSize);

  m_m.unlock();
}

int SimulationThread::cancel()
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  m_exit = true;
  m_m.unlock();
  return OpenThreads::Thread::cancel();
}

// Upload new boundary conditions
void SimulationThread::uploadBCs()
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  // k->uploadBCs();
  m_m.unlock();
}

// Reset the averaging array
void SimulationThread::resetAverages()
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  // k->resetAverages();
  m_m.unlock();
}

// Reset the simulation
void SimulationThread::resetDfs()
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  // k->initDomain(1.0, 0, 0, 0, k->Tinit);
  // simTimer->reset();
  m_m.unlock();
}

void SimulationThread::pause(bool state)
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  m_paused = state;
  m_m.unlock();
}

bool SimulationThread::isPaused()
{
  return m_paused;
}

// Redraw the visualization plot
void SimulationThread::draw(real *plot, DisplayQuantity::Enum visQ)
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
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
  m_m.unlock();

  cudaDeviceSynchronize();
}

void SimulationThread::run()
{
  while (true)
  {
    m_l.lock();
    m_n.lock();
    m_m.lock();
    m_n.unlock();
    if (!m_paused)
    {
      m_domainData->m_kernelData->compute(thrust::raw_pointer_cast(&(m_plot)[0]), m_visQ);
      m_domainData->m_simTimer->tick();
    }
    m_m.unlock();
    m_l.unlock();

    if (m_exit)
      return;

    // std::cout << *(m_domainData->m_simTimer) << std::endl;
    // std::cout << "MLUPS:" << m_domainData->m_simTimer->getMLUPS() << " RT rate:" << m_domainData->m_simTimer->getRealTimeRate() << std::endl;

    cudaDeviceSynchronize();
  }
}