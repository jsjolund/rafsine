#pragma once

#include <stdint.h>

#include <QMutex>
#include <QObject>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "AverageObserver.hpp"
#include "AveragingTimerCallback.hpp"
#include "BoundaryConditionTimerCallback.hpp"
#include "DomainData.hpp"
#include "LbmFile.hpp"
#include "SimulationTimer.hpp"
#include "Vector3.hpp"

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
  /**
   * @brief Construct a new Simulation Worker
   *
   * @param lbmFile LBM project file
   * @param nd Number of CUDA devices
   * @param avgPeriod Optional averaging period
   */
  explicit SimulationWorker(LbmFile lbmFile, int nd = 1, float avgPeriod = -1);
  ~SimulationWorker() { std::cout << "Destroying simulation" << std::endl; }
  /**
   * @brief Set max iterations to run before aborting simulation
   *
   * @param maxIterations
   */
  void setMaxIterations(unsigned int maxIterations) {
    m_maxIterations = maxIterations;
  }
  /**
   * @brief Add an observer for averaging
   *
   * @param observer
   */
  void addAveragingObserver(AverageObserver* observer);
  /**
   * @brief Set the averaging period in seconds
   *
   * @param seconds
   */
  inline void setAveragingPeriod(float seconds) {
    m_domain.m_avgPeriod = seconds;
    m_avgCallback->setTimeout(0);
    m_avgCallback->setRepeatTime(m_domain.m_avgPeriod);
    m_avgCallback->pause(seconds <= 0);
    m_domain.m_timer->addTimerCallback(m_avgCallback);
  }
  /**
   * @return std::shared_ptr<VoxelGeometry> Pointer to voxel geometry
   */
  inline std::shared_ptr<VoxelGeometry> getVoxels() {
    return m_domain.m_voxGeo;
  }
  /**
   * @return std::shared_ptr<BoundaryConditions> Pointer to array of boundary
   * conditions
   */
  inline std::shared_ptr<BoundaryConditions> getBoundaryConditions() {
    return m_domain.m_bcs;
  }
  /**
   * @return std::shared_ptr<SimulationTimer> Pointer to simulation timer
   */
  inline std::shared_ptr<SimulationTimer> getSimulationTimer() {
    return m_domain.m_timer;
  }
  /**
   * @return std::shared_ptr<UnitConverter>  Pointer to (real/lattice) unit
   * converter
   */
  inline std::shared_ptr<UnitConverter> getUnitConverter() {
    return m_domain.m_unitConverter;
  }
  /**
   * @return Vector3<size_t> Size of domain geometry
   */
  inline Vector3<size_t> getDomainSize() {
    return Vector3<size_t>(m_domain.m_nx, m_domain.m_ny, m_domain.m_nz);
  }

  /**
   * @return size_t 4:th dimension size
   */
  size_t getnd() { return m_domain.m_kernel->getnd(); }

  /**
   * @return std::string Path to voxel osg mesh
   */
  std::string getVoxelMeshPath() { return m_domain.getVoxelMeshPath(); }

  /**
   * @return D3Q4::Enum Multi-GPU partitioning axis
   */
  D3Q4::Enum getPartitioning() { return m_domain.m_kernel->getPartitioning(); }

  /**
   * @brief Get minimum and maximum from lattice histogram
   *
   * @param min
   * @param max
   * @param histogram
   */
  void getMinMax(real_t* min,
                 real_t* max,
                 thrust::host_vector<real_t>* histogram);

  /**
   * @brief Upload boundary conditions from host to device
   */
  void uploadBCs();

  /**
   * @brief Reset the simulation to initial condition
   */
  void resetDfs();

  /**
   * @brief Perform drawing of display slices
   *
   * @param visQ
   * @param slicePos
   * @param sliceX
   * @param sliceY
   * @param sliceZ
   */
  void draw(DisplayQuantity::Enum visQ,
            Vector3<int> slicePos,
            real_t* sliceX,
            real_t* sliceY,
            real_t* sliceZ);

  /**
   * @brief Stop the simulation
   * @return int Zero if success
   */
  int cancel();
  /**
   * @brief Resume the simulation
   * @return int Zero if success
   */
  int resume();

 public slots:
  /**
   * @brief QT slot for sumulation running
   *
   * @param iterations
   */
  void run(unsigned int iterations = 0);

 signals:
  /**
   * @brief QT signal for simulation finished
   *
   */
  void finished();
};
