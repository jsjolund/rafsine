#include "ConsoleClient.hpp"

void ConsoleClient::secUpdate() {
  SimulationTimer *timer = m_simWorker->getDomainData()->m_timer;
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
  m_simThread->start();
  std::cout << "Simulation is running..." << std::endl;
}

void ConsoleClient::close() {
  m_secTimer->stop();
  m_simWorker->cancel();
  m_simThread->quit();
  std::cout << "Waiting for simulation threads..." << std::endl;
  m_simThread->wait();
  emit finished();
}

ConsoleClient::ConsoleClient(LbmFile lbmFile, uint64_t iterations,
                             int numDevices, QObject *parent = 0)
    : QObject(parent), m_numDevices(numDevices) {
  m_simWorker = new SimulationWorker(lbmFile, iterations, numDevices);
  m_simThread = new QThread;
  m_simWorker->moveToThread(m_simThread);
  connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
  connect(m_simWorker, SIGNAL(finished()), this, SLOT(close()));

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));
}
