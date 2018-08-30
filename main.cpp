#include <QDesktopWidget>
#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include <cuda_profiler_api.h>

#include "DomainData.hpp"
#include "SimulationThread.hpp"
#include "MainWindow.hpp"

cudaStream_t simStream = 0;
cudaStream_t renderStream = 0;

int main(int argc, char **argv)
{
  Q_INIT_RESOURCE(res);
  QApplication app(argc, argv);
  QCoreApplication::setOrganizationName("RISE SICS North");
  QCoreApplication::setApplicationName("LUA LBM GPU Leeds 2013");
  QCoreApplication::setApplicationVersion("v0.1");
  QCommandLineParser parser;
  parser.setApplicationDescription(QCoreApplication::applicationName());
  parser.addHelpOption();
  parser.addVersionOption();
  QCommandLineOption settingsOpt({"s", "settings"}, "Lua LBM settings script.", "settings");
  QCommandLineOption geometryOpt({"g", "geometry"}, "Lua LBM geometry script.", "geometry");
  parser.addOption(settingsOpt);
  parser.addOption(geometryOpt);
  parser.process(app);

  QString settingsFileName = parser.value("settings");
  QString geometryFileName = parser.value("geometry");

  cudaProfilerStart();
  int priorityHigh, priorityLow;
  cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
  cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priorityHigh);
  // cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, priorityLow);

  DomainData *domainData = new DomainData();
  SimulationThread *simThread = new SimulationThread();
  if (parser.isSet(settingsOpt) && !settingsFileName.isEmpty() && parser.isSet(geometryOpt) && !geometryFileName.isEmpty())
  {
    domainData->loadFromLua(settingsFileName.toUtf8().constData(), geometryFileName.toUtf8().constData());
    simThread->setDomainData(domainData);
    simThread->start();
  }

  simThread->setSchedulePriority(OpenThreads::Thread::ThreadPriority ::THREAD_PRIORITY_MIN);

  MainWindow window(simThread);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
  QObject::connect(&app, SIGNAL(aboutToQuit()), &window, SLOT(closing()));

  const int retval = app.exec();
  QApplication::quit();

  cudaProfilerStop();
  simThread->cancel();
  simThread->join();
  cudaStreamSynchronize(0);
  cudaDeviceSynchronize();
  cudaDeviceReset();

  return retval;
}