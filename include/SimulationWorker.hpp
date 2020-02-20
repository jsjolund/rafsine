#pragma once

#include <stdint.h>

#include <QMutex>
#include <QObject>
#include <iostream>
#include <memory>
#include <vector>

#include "AverageObserver.hpp"
#include "AveragingTimerCallback.hpp"
#include "BoundaryConditionTimerCallback.hpp"
#include "DomainData.hpp"
#include "Eigen/Geometry"
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
  unsigned int m_maxIterations;
  //! The data of the problem domain to simulate
  DomainData m_domain;
  //! Simulation timer to perform averaging
  std::shared_ptr<AveragingTimerCallback> m_avgCallback;
  //! Simulation timer to update boundary conditions
  std::shared_ptr<BoundaryConditionTimerCallback> m_bcCallback;
  //! Visualization quantity
  DisplayQuantity::Enum m_displayQuantity;

 public:
  explicit SimulationWorker(LbmFile lbmFile,
                            int numDevices = 1,
                            float avgPeriod = -1);
  ~SimulationWorker() { std::cout << "Destroying simulation" << std::endl; }

  void setMaxIterations(unsigned int maxIterations) {
    m_maxIterations = maxIterations;
  }

  void addAveragingObserver(AverageObserver* observer);

  inline void setAveragingPeriod(float seconds) {
    m_domain.m_avgPeriod = seconds;
    m_avgCallback->setTimeout(0);
    m_avgCallback->setRepeatTime(m_domain.m_avgPeriod);
    m_avgCallback->pause(seconds <= 0);
    m_domain.m_timer->addTimerCallback(m_avgCallback);
  }

  inline std::shared_ptr<VoxelGeometry> getVoxels() {
    return m_domain.m_voxGeo;
  }
  inline std::shared_ptr<BoundaryConditions> getBoundaryConditions() {
    return m_domain.m_bcs;
  }
  inline std::shared_ptr<SimulationTimer> getSimulationTimer() {
    return m_domain.m_timer;
  }
  inline std::shared_ptr<UnitConverter> getUnitConverter() {
    return m_domain.m_unitConverter;
  }
  inline DomainData* getDomainData() { return &m_domain; }

  int getNumDevices() { return m_domain.m_kernel->getNumDevices(); }

  D3Q4::Enum getPartitioning() { return m_domain.m_kernel->getPartitioning(); }

  void getMinMax(real* min, real* max, thrust::host_vector<real>* histogram);

  // Upload new boundary conditions
  void uploadBCs();

  // Reset the simulation
  void resetDfs();

  void draw(DisplayQuantity::Enum visQ,
            Eigen::Vector3i slicePos,
            real* sliceX,
            real* sliceY,
            real* sliceZ);

  int cancel();
  int resume();

 public slots:
  void run(unsigned int iterations = 0);

 signals:
  void finished();
};
