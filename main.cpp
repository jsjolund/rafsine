#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QDesktopWidget>
#include <QObject>

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include <cuda_profiler_api.h>

#include "DomainData.hpp"
#include "MainWindow.hpp"
#include "SimulationWorker.hpp"

int main(int argc, char **argv) {
  Q_INIT_RESOURCE(res);
  QApplication app(argc, argv);
  QCoreApplication::setOrganizationName("RISE SICS North");
  QCoreApplication::setApplicationName("LUA LBM GPU Leeds Luleå 2018");
  QCoreApplication::setApplicationVersion("v0.1");
  QCommandLineParser parser;
  parser.setApplicationDescription(QCoreApplication::applicationName());
  parser.addHelpOption();
  parser.addVersionOption();
  QCommandLineOption settingsOpt({"s", "settings"}, "Lua LBM settings script.",
                                 "settings");
  QCommandLineOption geometryOpt({"g", "geometry"}, "Lua LBM geometry script.",
                                 "geometry");
  parser.addOption(settingsOpt);
  parser.addOption(geometryOpt);
  parser.process(app);

  QString settingsFilePath = parser.value("settings");
  QString geometryFilePath = parser.value("geometry");

  CUDA_RT_CALL(cudaProfilerStart());
  CUDA_RT_CALL(cudaSetDevice(0));
  int numDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
  numDevices = min(numDevices, 8);

  DomainData *domainData = new DomainData(numDevices);
  SimulationWorker *simWorker = new SimulationWorker();
  if (!settingsFilePath.isEmpty() && !geometryFilePath.isEmpty()) {
    domainData->loadFromLua(geometryFilePath.toUtf8().constData(),
                            settingsFilePath.toUtf8().constData());
    simWorker->setDomainData(domainData);
  }

  MainWindow window(simWorker, numDevices);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);

  const int retval = app.exec();

  CUDA_RT_CALL(cudaProfilerStop());

#pragma omp parallel num_threads(numDevices)
  {
    const int dev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaDeviceReset());
  }

  std::cout << "Exited" << std::endl;

  return retval;
}
