#pragma once

#include <QObject>
#include <QSocketNotifier>
#include <QTextStream>
#include <QThread>
#include <QTimer>

#include <stdio.h>
#include <unistd.h>  //Provides STDIN_FILENO
#include <iostream>

#include "DomainData.hpp"
#include "SimulationWorker.hpp"

class ConsoleReader : public QObject {
  Q_OBJECT

 public:
  explicit ConsoleReader(QObject *parent = 0);

 signals:
  void textReceived(QString message);

 public slots:
  void text();

 private:
  QSocketNotifier m_notifier;
};

class ConsoleClient : public QObject {
  Q_OBJECT

 private:
  // Number of GPUs to use
  int m_numDevices;
  // Pointer to the currently running simulation thread
  QThread *m_simThread;
  // Worker object for the simulation thread
  SimulationWorker *m_simWorker;

  ConsoleReader *m_reader;

  QTimer *m_secTimer;

  Q_SLOT void secUpdate();

 signals:
  void finished();

 protected:
 public slots:
  void run();
  void close();
  void handleInput(QString message);

 public:
  ConsoleClient(SimulationWorker *simWorker, int numDevices, QObject *parent);
};
