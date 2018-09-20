#pragma once

#include <mutex>
#include <iostream>

#include <QObject>
#include <QMutex>

#include <osg/Vec3i>
#include <osg/ref_ptr>

#include "DomainData.hpp"
#include "SimulationTimer.hpp"

#define SIM_HIGH_PRIO_LOCK \
  m_n.lock();              \
  m_m.lock();              \
  m_n.unlock();
#define SIM_HIGH_PRIO_UNLOCK m_m.unlock();
#define SIM_LOW_PRIO_LOCK \
  m_l.lock();             \
  m_n.lock();             \
  m_m.lock();             \
  m_n.unlock();
#define SIM_LOW_PRIO_UNLOCK \
  m_m.unlock();             \
  m_l.unlock();

class SimulationWorker : public QObject
{
  Q_OBJECT

private:
  // Quantity to be visualised on plot
  DisplayQuantity::Enum m_visQ;
  // Triple mutex for prioritized access
  QMutex m_l, m_m, m_n;
  // Buffer for OpenGL plot, copied when drawing is requested
  thrust::device_vector<real> m_plot;
  // Signals exit of simulation loop
  volatile bool m_exit;

  DomainData *m_domainData;

public:
  SimulationWorker(DomainData *domainData);
  SimulationWorker();
  // ~SimulationWorker();

  inline std::shared_ptr<VoxelGeometry> getVoxelGeometry()
  {
    return m_domainData->m_voxGeo;
  }
  void setDomainData(DomainData *m_domainData);
  DomainData *getDomainData();
  bool hasDomainData();

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the averaging array
  void resetAverages();

  // Reset the simulation
  void resetDfs();

  void draw(real *plot, DisplayQuantity::Enum visQ);

  int cancel();
  int resume();

public slots:
  void run();

signals:
  void finished();
};