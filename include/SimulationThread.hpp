#pragma once

#include <mutex>
#include <iostream>

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <osg/Vec3i>
#include <osg/ref_ptr>

#include "DomainData.hpp"
#include "SimulationTimer.hpp"

class SimulationThread : public OpenThreads::Thread
{
private:
  // Quantity to be visualised on plot
  DisplayQuantity::Enum m_visQ;
  // Triple mutex for prioritized access
  OpenThreads::Mutex m_l, m_m, m_n;
  // Buffer for OpenGL plot, copied when drawing is requested
  thrust::device_vector<real> m_plot;
  // Signals exit of simulation loop
  volatile bool m_exit;
  // Signals simulation pause
  volatile bool m_paused;

  DomainData *m_domainData;

public:
  SimulationThread(DomainData *domainData);
  SimulationThread();
  ~SimulationThread();

  inline std::shared_ptr<VoxelGeometry> getVoxelGeometry()
  {
    return m_domainData->m_voxGeo;
  }

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the averaging array
  void resetAverages();

  // Reset the simulation
  void resetDfs();

  void pause(bool state);
  bool isPaused();
  void draw(real *plot, DisplayQuantity::Enum visQ);

  virtual void run();

  virtual int cancel();
};
