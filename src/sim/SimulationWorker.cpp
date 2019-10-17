#include "SimulationWorker.hpp"

SimulationWorker::SimulationWorker(LbmFile lbmFile,
                                   int numDevices,
                                   float avgPeriod)
    : m_domain(),
      m_exit(false),
      m_visQ(DisplayQuantity::TEMPERATURE),
      m_maxIterations(0) {
  m_domain.loadFromLua(numDevices, lbmFile.getGeometryPath(),
                       lbmFile.getSettingsPath());
  // Reset the simulation timer
  m_domain.m_timer->reset();
  m_domain.m_timer->setTime(lbmFile.getStartTime());

  // This timer will set the boundary conditions according to the input csv file
  m_bcCallback = std::make_shared<BoundaryConditionTimerCallback>(
      m_domain.m_kernel, m_domain.m_bcs, m_domain.m_voxGeo,
      m_domain.m_unitConverter, lbmFile.getInputCSVPath());
  m_bcCallback->setTimeout(0);
  m_domain.m_timer->addSimulationTimer(m_bcCallback);

  // This timer will read the averaging array periodically
  if (avgPeriod > 0) m_domain.m_avgPeriod = avgPeriod;
  m_avgCallback = std::make_shared<AveragingTimerCallback>(
      m_domain.m_kernel, m_domain.m_unitConverter,
      *m_domain.m_voxGeo->getSensors());
  m_avgCallback->setTimeout(0);
  m_avgCallback->setRepeatTime(m_domain.m_avgPeriod);
  m_avgCallback->pause(m_domain.m_avgPeriod <= 0);
  m_domain.m_timer->addSimulationTimer(m_avgCallback);
}

void SimulationWorker::addAveragingObserver(AverageObserver* observer) {
  if (m_domain.m_avgPeriod <= 0.0)
    throw std::runtime_error(ErrorFormat() << "Invalid averaging period "
                                           << m_domain.m_avgPeriod);
  m_avgCallback->addObserver(observer);
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

void SimulationWorker::uploadBCs() {
  SIM_HIGH_PRIO_LOCK();
  m_domain.m_kernel->uploadBCs(m_domain.m_bcs);
  SIM_HIGH_PRIO_UNLOCK();
}

void SimulationWorker::resetDfs() {
  SIM_HIGH_PRIO_LOCK();
  // Reset simulation timer and averaging callback
  m_domain.m_timer->reset();

  m_bcCallback->reset();
  m_bcCallback->setTimeout(0);
  m_domain.m_timer->addSimulationTimer(m_bcCallback);

  m_avgCallback->reset();
  m_avgCallback->setTimeout(0);
  m_avgCallback->setRepeatTime(m_domain.m_avgPeriod);
  m_avgCallback->pause(m_domain.m_avgPeriod <= 0);
  m_domain.m_timer->addSimulationTimer(m_avgCallback);

  // Reset the averaging array on next kernel execution
  m_domain.m_kernel->resetAverages();
  // Set the distribution functions to initial state
  m_domain.m_kernel->resetDfs();
  SIM_HIGH_PRIO_UNLOCK();
}

void SimulationWorker::getMinMax(real* min, real* max) {
  SIM_HIGH_PRIO_LOCK();
  m_domain.m_kernel->getMinMax(min, max);
  SIM_HIGH_PRIO_UNLOCK();
}

void SimulationWorker::draw(DisplayQuantity::Enum visQ,
                            Eigen::Vector3i slicePos,
                            real* sliceX,
                            real* sliceY,
                            real* sliceZ) {
  if (!m_exit) {
    SIM_HIGH_PRIO_LOCK();
    // Since the LBM kernel only draws one of the display quantities, we may
    // need to run the kernel again to update the plot (back)buffer
    if (m_visQ != visQ) {
      m_visQ = visQ;
      m_domain.m_kernel->compute(m_visQ);
      m_domain.m_timer->tick();
    }
    // Here the actual drawing takes place
    m_domain.m_kernel->compute(m_visQ, slicePos, sliceX, sliceY, sliceZ);
    m_domain.m_timer->tick();
    SIM_HIGH_PRIO_UNLOCK();
  } else {
    // If simulation is paused, do only the drawing, do not increment timer
    SIM_HIGH_PRIO_LOCK();
    m_domain.m_kernel->compute(m_visQ, slicePos, sliceX, sliceY, sliceZ, false);
    SIM_HIGH_PRIO_UNLOCK();
  }
}

void SimulationWorker::run(const unsigned int iterations) {
  m_maxIterations = max(m_maxIterations, iterations);
  int i = 0;
  while (!m_exit && (m_maxIterations == 0 || i++ < m_maxIterations)) {
    SIM_LOW_PRIO_LOCK();
    m_domain.m_kernel->compute();
    m_domain.m_timer->tick();
    SIM_LOW_PRIO_UNLOCK();
  }
  emit finished();
}
