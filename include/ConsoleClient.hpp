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
  //! Pointer to the currently running simulation thread
  QThread* m_simThread;
  //! Worker object for the simulation thread
  SimulationWorker* m_simWorker;
  //! Flag to prevent multiple closings
  bool m_closing;
  // Repeating timer to print out stats
  QTimer* m_secTimer;
  //! Fake visualization
  bool m_visualize;
  //! Fake visualization timer
  QTimer* m_renderTimer;
  //! Fake visualization slices
  thrust::device_vector<real>* m_sliceX;
  thrust::device_vector<real>* m_sliceY;
  thrust::device_vector<real>* m_sliceZ;

 signals:
  void finished();

 public slots:
  void run();
  void close();
  void secUpdate();
  void render();

 public:
  ConsoleClient(LbmFile lbmFile,
                int numDevices,
                const unsigned int iterations,
                QObject* parent = 0,
                bool visualize = false);
};
