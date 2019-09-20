#pragma once

#include <QMutex>
#include <QObject>

#include <osg/Vec3i>
#include <osg/ref_ptr>

#include <stdint.h>
#include <iostream>
#include <memory>
#include <vector>

#include "AverageObserver.hpp"
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
  std::shared_ptr<AveragingTimerCallback> m_avgCallback;
  //! Observers for averaging
  std::vector<AverageObserver *> m_avgObservers;
  //! Simulation timer to update boundary conditions
  std::shared_ptr<BoundaryConditionTimerCallback> m_bcCallback;
  //! Visualization quantity
  DisplayQuantity::Enum m_visQ;

  bool abortSignalled();

 public:
  explicit SimulationWorker(LbmFile lbmFile, uint64_t maxIterations = 0,
                            int numDevices = 1);
  ~SimulationWorker() { std::cout << "Destroying simulation" << std::endl; }

  inline void addAverageingObserver(AverageObserver *observer) {
    m_avgObservers.push_back(observer);
    m_avgCallback->addObserver(*observer);
  }
  inline std::shared_ptr<VoxelGeometry> getVoxelGeometry() {
    return m_domain.m_voxGeo;
  }
  inline std::shared_ptr<UnitConverter> getUnitConverter() {
    return m_domain.m_unitConverter;
  }
  inline DomainData *getDomainData() { return &m_domain; }

  int getNumDevices() { return m_domain.m_kernel->getNumDevices(); }

  void getMinMax(real *min, real *max);

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the simulation
  void resetDfs();

  void draw(DisplayQuantity::Enum visQ, glm::ivec3 slicePos, real *sliceX,
            real *sliceY, real *sliceZ);

  int cancel();
  int resume();

 public slots:
  void run();

 signals:
  void finished();
};
