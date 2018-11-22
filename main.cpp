#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QDesktopWidget>
#include <QObject>

#include <stdio.h>
#include <unistd.h>
#include <iostream>

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
                                 "settings.lua");
  QCommandLineOption geometryOpt({"g", "geometry"}, "Lua LBM geometry script.",
                                 "geometry.lua");
  QCommandLineOption devicesOpt({"d", "devices"},
                                "Number of CUDA devices to use.", "1");
  parser.addOption(settingsOpt);
  parser.addOption(geometryOpt);
  parser.addOption(devicesOpt);
  parser.process(app);

  QString settingsFilePath = parser.value("settings");
  QString geometryFilePath = parser.value("geometry");

  int numSupportedDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numSupportedDevices));
  numSupportedDevices = min(8, numSupportedDevices);
  int numRequestedDevices = parser.value("devices").toInt();
  int numDevices =
      (numRequestedDevices == 0) ? numSupportedDevices : numRequestedDevices;

  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));
  int cudaDev;
  CUDA_RT_CALL(cudaGetDevice(&cudaDev));
  std::cout << "Using device " << cudaDev << std::endl;

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
