#include "ConsoleClient.hpp"

void ConsoleClient::render() {
  real_t* sliceXPtr = thrust::raw_pointer_cast(&(*m_sliceX)[0]);
  real_t* sliceYPtr = thrust::raw_pointer_cast(&(*m_sliceY)[0]);
  real_t* sliceZPtr = thrust::raw_pointer_cast(&(*m_sliceZ)[0]);

  if (!m_closing) {
    m_simWorker->draw(DisplayQuantity::TEMPERATURE, Vector3<int>(1, 1, 1),
                      sliceXPtr, sliceYPtr, sliceZPtr);
  }
}

void ConsoleClient::secUpdate() {
  std::shared_ptr<SimulationTimer> timer = m_simWorker->getSimulationTimer();
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
    std::shared_ptr<SimulationTimer> timer = m_simWorker->getSimulationTimer();
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

ConsoleClient::ConsoleClient(LbmFile lbmFile,
                             int nd,
                             const unsigned int iterations,
                             QObject* parent,
                             bool visualize)
    : QObject(parent), m_visualize(visualize) {
  m_simWorker = new SimulationWorker(lbmFile, nd);
  if (lbmFile.getOutputCSVPath().length() > 0)
    m_simWorker->addAveragingObserver(
        new CSVAveraging(lbmFile.getOutputCSVPath()));
  // m_simWorker->addAveragingObserver(new StdoutAveraging());
  m_simWorker->setMaxIterations(iterations);

  m_simThread = new QThread;
  m_simWorker->moveToThread(m_simThread);
  connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
  connect(m_simWorker, SIGNAL(finished()), this, SLOT(close()));

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));

  // Mock visualization to check performance
  if (m_visualize) {
    Vector3<size_t> n = m_simWorker->getDomainSize();
    m_sliceX = new thrust::device_vector<real_t>(n.y() * n.z());
    m_sliceY = new thrust::device_vector<real_t>(n.x() * n.z());
    m_sliceZ = new thrust::device_vector<real_t>(n.x() * n.y());
    m_renderTimer = new QTimer(this);
    connect(m_renderTimer, SIGNAL(timeout()), this, SLOT(render()));
  }
}
