#include "ConsoleClient.hpp"

ConsoleReader::ConsoleReader(QObject *parent)
    : QObject(parent), m_notifier(STDIN_FILENO, QSocketNotifier::Read) {
  connect(&m_notifier, SIGNAL(activated(int)), this, SLOT(text()));
}

void ConsoleReader::text() {
  QTextStream qin(stdin);
  QString line = qin.read(1);
  emit textReceived(line);
}

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

void ConsoleClient::handleInput(QString message) {
  std::cout << message.toUtf8().constData() << std::endl;
  if (message == "q") close();
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

  m_reader = new ConsoleReader(this);
  connect(m_reader, &ConsoleReader::textReceived, this,
          &ConsoleClient::handleInput);
}