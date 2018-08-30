#include <QApplication>
#include <QDesktopWidget>

#include <QCommandLineParser>
#include <QCommandLineOption>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include <cuda_profiler_api.h>

#include "DomainData.hpp"
#include "SimulationThread.hpp"
#include "CFDWidget.hpp"
#include "MainWindow.hpp"

SimulationThread *simThread;
cudaStream_t simStream = 0;
cudaStream_t renderStream = 0;


int main(int argc, char **argv)
{
  cudaProfilerStart();
  // CUDA stream priorities. Simulation has highest priority, rendering lowest.
  int priorityHigh, priorityLow;
  cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
  cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priorityHigh);
  // cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, priorityLow);

  DomainData *domainData = new DomainData();
  simThread = new SimulationThread(domainData);
  simThread->setSchedulePriority(OpenThreads::Thread::ThreadPriority ::THREAD_PRIORITY_MIN);

  QApplication app(argc, argv);

  MainWindow window;
  CFDWidget *widget = new CFDWidget(simThread, 1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
  widget->setFocus();

  simThread->start();

  QObject::connect(&app, SIGNAL(aboutToQuit()), &window, SLOT(closing()));
  const int retval = app.exec();

  cudaProfilerStop();
  simThread->cancel();
  simThread->join();
  cudaStreamSynchronize(0);
  cudaDeviceSynchronize();
  cudaDeviceReset();

  return retval;

  // Q_INIT_RESOURCE(application);

  // QApplication app(argc, argv);
  // QCoreApplication::setOrganizationName("QtProject");
  // QCoreApplication::setApplicationName("Application Example");
  // QCoreApplication::setApplicationVersion(QT_VERSION_STR);
  // QCommandLineParser parser;
  // parser.setApplicationDescription(QCoreApplication::applicationName());
  // parser.addHelpOption();
  // parser.addVersionOption();
  // parser.addPositionalArgument("file", "The file to open.");
  // parser.process(app);

  // MainWindow mainWin;
  // if (!parser.positionalArguments().isEmpty())
  //   mainWin.loadFile(parser.positionalArguments().first());
  // mainWin.show();
  // return app.exec();
}