#include "SimulationWorker.hpp"

SimulationWorker::SimulationWorker(LbmFile lbmFile, uint64_t maxIterations,
                                   int numDevices)
    : m_domain(numDevices),
      m_exit(false),
      m_maxIterations(maxIterations),
      m_visQ(DisplayQuantity::TEMPERATURE) {
  m_domain.loadFromLua(lbmFile.getGeometryPath(), lbmFile.getSettingsPath());
  // Reset the simulation timer
  m_domain.m_timer->reset();

  // This timer will set the boundary conditions according to the input csv file
  m_bcCallback = new BoundaryConditionTimerCallback(
      m_domain.m_kernel, m_domain.m_bcs, m_domain.m_voxGeo,
      m_domain.m_unitConverter, lbmFile.getInputCSVPath());
  m_domain.m_timer->setSimulationTime(lbmFile.getStartTime());
  m_bcCallback->setTimeout(0);
  m_domain.m_timer->addSimulationTimer(m_bcCallback);

  // This timer will read the averaging array periodically
  if (m_domain.m_avgPeriod > 0.0) {
    m_avgCallback = new AveragingTimerCallback(
        m_domain.m_kernel, m_domain.m_unitConverter,
        *m_domain.m_voxGeo->getSensors(), lbmFile.getOutputCSVPath());
    m_avgCallback->setTimeout(0);
    m_avgCallback->setRepeatTime(m_domain.m_avgPeriod);
    m_domain.m_timer->addSimulationTimer(m_avgCallback);
  }
}

int SimulationWorker::cancel() {
  SIM_HIGH_PRIO_LOCK();
  m_exit = true;
  SIM_HIGH_PRIO_UNLOCK();
  return 0;
}

int SimulationWorker::resume() {
  SIM_HIGH_PRIO_LOCK();
  m_exit = false;
  SIM_HIGH_PRIO_UNLOCK();
  return 0;
}

// Upload new boundary conditions
void SimulationWorker::uploadBCs() {
  SIM_HIGH_PRIO_LOCK();
  m_domain.m_kernel->uploadBCs(m_domain.m_bcs);
  SIM_HIGH_PRIO_UNLOCK();
}

// Reset the simulation
void SimulationWorker::resetDfs() {
  SIM_HIGH_PRIO_LOCK();
  // Reset simulation timer and averaging callback
  m_domain.m_timer->reset();
  if (m_domain.m_avgPeriod > 0.0) {
    m_avgCallback->setTimeout(m_domain.m_avgPeriod);
    m_avgCallback->setRepeatTime(m_domain.m_avgPeriod);
    m_avgCallback->m_lastTicks = 0;
    m_domain.m_timer->addSimulationTimer(m_avgCallback);
  }
  // Reset the averaging array on next kernel execution
  m_domain.m_kernel->resetAverages();
  // Set the distribution functions to initial state
  m_domain.m_kernel->resetDfs();
  SIM_HIGH_PRIO_UNLOCK();
}

bool SimulationWorker::abortSignalled() {
  return m_exit || (m_maxIterations > 0 &&
                    m_domain.m_timer->getTicks() >= m_maxIterations);
}

void SimulationWorker::getMinMax(real *min, real *max) {
  SIM_HIGH_PRIO_LOCK();
  m_domain.m_kernel->getMinMax(min, max);
  SIM_HIGH_PRIO_UNLOCK();
}

// Draw the visualization plot
void SimulationWorker::draw(DisplayQuantity::Enum visQ, glm::ivec3 slicePos,
                            real *sliceX, real *sliceY, real *sliceZ) {
  if (!abortSignalled()) {
    SIM_HIGH_PRIO_LOCK();
    if (m_visQ != visQ) {
      m_visQ = visQ;
      m_domain.m_timer->tick();
      m_domain.m_kernel->compute(m_visQ);
    }
    m_domain.m_timer->tick();
    m_domain.m_kernel->compute(m_visQ, slicePos, sliceX, sliceY, sliceZ);
    SIM_HIGH_PRIO_UNLOCK();
  }
}

void SimulationWorker::run() {
  while (!abortSignalled()) {
    SIM_LOW_PRIO_LOCK();
    m_domain.m_timer->tick();
    m_domain.m_kernel->compute(m_visQ);
    SIM_LOW_PRIO_UNLOCK();
  }
  emit finished();
}
