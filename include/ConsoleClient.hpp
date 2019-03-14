#pragma once

#include <QObject>
#include <QSocketNotifier>
#include <QTextStream>
#include <QThread>
#include <QTimer>

#include <stdio.h>
#include <iostream>

#include "DomainData.hpp"
#include "SimulationWorker.hpp"

class ConsoleClient : public QObject {
  Q_OBJECT

 private:
  // Number of GPUs to use
  int m_numDevices;
  // Pointer to the currently running simulation thread
  QThread *m_simThread;
  // Worker object for the simulation thread
  SimulationWorker *m_simWorker;

  QTimer *m_secTimer;

 signals:
  void finished();

 protected:
 public slots:
  void run();
  void close();
  void secUpdate();

 public:
  ConsoleClient(LbmFile lbmFile, uint64_t iterations, int numDevices,
                QObject *parent);
};
