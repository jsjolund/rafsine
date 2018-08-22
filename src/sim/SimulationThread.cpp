#include "SimulationThread.hpp"

SimulationThread::SimulationThread()
    : MyThreadClass(),
      m_paused(false),
      m_exit(false),
      m_time(0),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE)
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();

  m_domainData = new DomainData();
  m_mesh = new VoxelMesh(*(m_domainData->m_voxGeo->data));
  osg::Vec3i voxSize = osg::Vec3i(m_mesh->getSizeX(),
                                  m_mesh->getSizeY(),
                                  m_mesh->getSizeZ());
  osg::Vec3i voxMin = osg::Vec3i(-1, -1, -1);
  osg::Vec3i voxMax = osg::Vec3i(voxSize);
  m_plot = thrust::device_vector<real>(m_mesh->getSizeX() * m_mesh->getSizeY() * m_mesh->getSizeZ());
  m_mesh->buildMesh(voxMin, voxMax);

  m_m.unlock();
}

void SimulationThread::cancel()
{
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  m_exit = true;
  m_m.unlock();
  // return OpenThreads::Thread::cancel();
}

osg::ref_ptr<VoxelMesh> SimulationThread::getVoxelMesh()
{
  osg::ref_ptr<VoxelMesh> meshPtr;
  m_n.lock();
  m_m.lock();
  m_n.unlock();
  meshPtr = m_mesh;
  m_m.unlock();
  return meshPtr;
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
    m_time++;
  }
  int size = m_mesh->getSizeX() * m_mesh->getSizeY() * m_mesh->getSizeZ();
  thrust::device_ptr<real> dp1(thrust::raw_pointer_cast(&(m_plot)[0]));
  thrust::device_ptr<real> dp2(plot);
  thrust::copy(dp1, dp1 + size, dp2);
  m_m.unlock();
}

void SimulationThread::InternalThreadEntry()
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
      m_time++;
      // simTimer->update();
    }
    m_m.unlock();
    m_l.unlock();

    if (m_exit)
      return;
    usleep(1000);
    // pthread_yield();
  }
}