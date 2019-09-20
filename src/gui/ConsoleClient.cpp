#include "ConsoleClient.hpp"

void ConsoleClient::render() {
  real *sliceXPtr = thrust::raw_pointer_cast(&(*m_sliceX)[0]);
  real *sliceYPtr = thrust::raw_pointer_cast(&(*m_sliceY)[0]);
  real *sliceZPtr = thrust::raw_pointer_cast(&(*m_sliceZ)[0]);

  if (!m_closing) {
    m_simWorker->draw(DisplayQuantity::TEMPERATURE, glm::ivec3(1, 1, 1),
                      sliceXPtr, sliceYPtr, sliceZPtr);
  }
}

void ConsoleClient::secUpdate() {
  std::shared_ptr<SimulationTimer> timer =
      m_simWorker->getDomainData()->m_timer;
  std::ostringstream stream;
  stream << '\r';
  stream << "Time: " << *timer;
  stream << ", Rate: " << timer->getRealTimeRate();
  stream << ", MLUPS: " << timer->getMLUPS();
  stream << ", LUPS: " << timer->getLUPS();
  std::cout << stream.str() << std::flush;
}

void ConsoleClient::run() {
  m_secTimer->start(1000);
  if (m_visualize) m_renderTimer->start(17);
  m_simThread->start();
  std::cout << "Simulation is running..." << std::endl;
}

void ConsoleClient::close() {
  if (!m_closing) {
    std::shared_ptr<SimulationTimer> timer =
        m_simWorker->getDomainData()->m_timer;
    std::cout << std::endl;
    std::cout << "Average MLUPS: " << timer->getAverageMLUPS() << std::endl;

    m_closing = true;
    m_secTimer->stop();
    if (m_visualize) m_renderTimer->stop();
    m_simWorker->cancel();
    m_simThread->quit();
    std::cout << "Waiting for simulation threads..." << std::endl;
    m_simThread->wait();
    if (m_visualize) {
      delete m_sliceX;
      delete m_sliceY;
      delete m_sliceZ;
    }
    emit finished();
  }
}

ConsoleClient::ConsoleClient(LbmFile lbmFile, uint64_t iterations,
                             int numDevices, QObject *parent, bool visualize)
    : QObject(parent), m_visualize(visualize) {
  m_simWorker = new SimulationWorker(lbmFile, iterations, numDevices);

  m_simThread = new QThread;
  m_simWorker->moveToThread(m_simThread);
  connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
  connect(m_simWorker, SIGNAL(finished()), this, SLOT(close()));

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));

  if (m_visualize) {
    glm::ivec3 n = m_simWorker->getDomainData()->m_kernel->getDims();
    m_sliceX = new thrust::device_vector<real>(n.y * n.z);
    m_sliceY = new thrust::device_vector<real>(n.x * n.z);
    m_sliceZ = new thrust::device_vector<real>(n.x * n.y);
    m_renderTimer = new QTimer(this);
    connect(m_renderTimer, SIGNAL(timeout()), this, SLOT(render()));
  }
}
