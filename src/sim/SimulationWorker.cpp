#include "SimulationWorker.hpp"

SimulationWorker::~SimulationWorker() { delete m_domain; }

SimulationWorker::SimulationWorker(DomainData *domainData,
                                   uint64_t maxIterations)
    : m_domain(domainData),
      m_exit(false),
      m_maxIterations(maxIterations),
      m_visQ(DisplayQuantity::Enum::TEMPERATURE) {
  setDomainData(domainData);
}

void SimulationWorker::setDomainData(DomainData *domainData) {
  if (!domainData) return;
  SIM_HIGH_PRIO_LOCK();
  if (m_domain) delete m_domain;
  m_domain = domainData;
  SIM_HIGH_PRIO_UNLOCK();
}

bool SimulationWorker::hasDomainData() { return m_domain != NULL; }

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
  m_domain->m_kernel->uploadBCs(m_domain->m_bcs);
  SIM_HIGH_PRIO_UNLOCK();
}

// Reset the averaging array
void SimulationWorker::resetAverages() {
  if (!m_domain) return;
  SIM_HIGH_PRIO_LOCK();
  m_domain->m_kernel->resetAverages();
  SIM_HIGH_PRIO_UNLOCK();
}

// Reset the simulation
void SimulationWorker::resetDfs() {
  if (!m_domain) return;
  SIM_HIGH_PRIO_LOCK();
  m_domain->m_timer->reset();
  m_domain->m_kernel->resetAverages();
  m_domain->m_kernel->resetDfs();
  SIM_HIGH_PRIO_UNLOCK();
}

void SimulationWorker::runKernel() {
  // m_domain->m_kernel->compute(thrust::raw_pointer_cast(&(m_plot)[0]),
  // m_visQ);
  m_domain->m_kernel->compute(m_visQ);
  m_domain->m_timer->tick();
}

bool SimulationWorker::abortSignalled() {
  return m_exit || (m_maxIterations > 0 &&
                    m_domain->m_timer->getTicks() >= m_maxIterations);
}

// Redraw the visualization plot
void SimulationWorker::draw(thrust::device_vector<real> *plot,
                            DisplayQuantity::Enum visQ) {
  if (visQ != m_visQ) {
    SIM_HIGH_PRIO_LOCK();
    m_visQ = visQ;
    if (!abortSignalled()) runKernel();
    SIM_HIGH_PRIO_UNLOCK();
  }
  m_domain->m_kernel->plot(0, plot);
}

void SimulationWorker::run() {
  while (!abortSignalled()) {
    SIM_LOW_PRIO_LOCK();
    runKernel();
    SIM_LOW_PRIO_UNLOCK();
  }
  emit finished();
}
