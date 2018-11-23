#include "ConsoleClient.hpp"

void ConsoleClient::secUpdate() {
  if (m_simWorker->hasDomainData()) {
    SimulationTimer *simTimer = m_simWorker->getDomainData()->m_simTimer;
    std::ostringstream stream;
    stream << '\r';
    stream << "Time: " << *simTimer;
    stream << ", Rate: " << simTimer->getRealTimeRate();
    stream << ", MLUPS: " << simTimer->getMLUPS();
    stream << ", LUPS: " << simTimer->getLUPS();
    std::cout << stream.str() << std::flush;
  }
}

void ConsoleClient::run() {
  if (!m_simWorker->hasDomainData()) emit finished();

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

ConsoleClient::ConsoleClient(SimulationWorker *simWorker, int numDevices,
                             QObject *parent = 0)
    : QObject(parent), m_numDevices(numDevices), m_simWorker(simWorker) {
  // Simulation thread
  m_simThread = new QThread;
  m_simWorker->moveToThread(m_simThread);
  connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
  connect(m_simWorker, SIGNAL(finished()), m_simThread, SLOT(quit()));

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));
}