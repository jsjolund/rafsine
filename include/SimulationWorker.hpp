#pragma once

#include <QMutex>
#include <QObject>

#include <osg/Vec3i>
#include <osg/ref_ptr>

#include <stdint.h>
#include <iostream>
#include <memory>

#include "AveragingTimerCallback.hpp"
#include "BoundaryConditionTimerCallback.hpp"
#include "DomainData.hpp"
#include "LbmFile.hpp"
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
  //! Triple mutex for prioritized access
  QMutex m_l, m_m, m_n;
  //! Signals the exit of simulation loop
  volatile bool m_exit;
  //! Number of simulation steps to run before exiting (0 for infinite)
  const uint64_t m_maxIterations;
  //! The data of the problem domain to simulate
  DomainData m_domain;
  //! Simulation timer to perform averaging
  AveragingTimerCallback *m_avgCallback;
  //! Simulation timer to update boundary conditions
  BoundaryConditionTimerCallback *m_bcCallback;

  bool abortSignalled();

 public:
  explicit SimulationWorker(LbmFile lbmFile, uint64_t maxIterations = 0,
                            int numDevices = 1, bool plotEnabled = true);
  ~SimulationWorker() {
    std::cout << "Destroying simulation" << std::endl;
    delete m_avgCallback;
    delete m_bcCallback;
  }

  inline std::shared_ptr<VoxelGeometry> getVoxelGeometry() {
    return m_domain.m_voxGeo;
  }
  inline std::shared_ptr<UnitConverter> getUnitConverter() {
    return m_domain.m_unitConverter;
  }
  inline DomainData *getDomainData() { return &m_domain; }

  void getMinMax(real *min, real *max);

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the simulation
  void resetDfs();

  void draw(DisplayQuantity::Enum visQ, glm::ivec3 slicePos, real *plotX,
            real *plotY, real *plotZ);

  int cancel();
  int resume();

 public slots:
  void run();

 signals:
  void finished();
};
