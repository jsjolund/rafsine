#pragma once

#include <QMutex>
#include <QObject>

#include <osg/Vec3i>
#include <osg/ref_ptr>

#include <stdint.h>
#include <iostream>

#include "DomainData.hpp"
#include "SimulationTimer.hpp"

#define SIM_HIGH_PRIO_LOCK() \
  {                          \
    m_n.lock();              \
    m_m.lock();              \
    m_n.unlock();            \
  }
#define SIM_HIGH_PRIO_UNLOCK() \
  { m_m.unlock(); }
#define SIM_LOW_PRIO_LOCK() \
  {                         \
    m_l.lock();             \
    m_n.lock();             \
    m_m.lock();             \
    m_n.unlock();           \
  }
#define SIM_LOW_PRIO_UNLOCK() \
  {                           \
    m_m.unlock();             \
    m_l.unlock();             \
  }

/**
 * @brief Worker class for the simulation execution thread. Uses a triple mutex
 * system for low and high priority tasks. Simulation is executed at low
 * priority, while pausing, resuming, resetting, updating etc is high priority.
 *
 */
class SimulationWorker : public QObject {
  Q_OBJECT

 private:
  // Quantity to be visualised on plot
  DisplayQuantity::Enum m_visQ;
  // Triple mutex for prioritized access
  QMutex m_l, m_m, m_n;
  // Signals exit of simulation loop
  volatile bool m_exit;
  const uint64_t m_maxIterations;

  DomainData *m_domain;

  bool abortSignalled();

 public:
  explicit SimulationWorker(DomainData *domainData = NULL,
                            uint64_t maxIterations = 0);
  ~SimulationWorker();

  inline std::shared_ptr<VoxelGeometry> getVoxelGeometry() {
    return m_domain->m_voxGeo;
  }
  inline std::shared_ptr<UnitConverter> getUnitConverter() {
    return m_domain->m_unitConverter;
  }
  inline DomainData *getDomainData() { return m_domain; }
  void setDomainData(DomainData *m_domain);

  bool hasDomainData();

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the averaging array
  void resetAverages();

  // Reset the simulation
  void resetDfs();

  void draw(thrust::device_vector<real> *plot, DisplayQuantity::Enum visQ,
            glm::ivec3 slicePos);

  int cancel();
  int resume();

 public slots:
  void run();

 signals:
  void finished();
};
