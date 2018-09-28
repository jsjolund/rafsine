#include <QDesktopWidget>
#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QObject>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include <cuda_profiler_api.h>

#include "DomainData.hpp"
#include "SimulationWorker.hpp"
#include "MainWindow.hpp"

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

  QString settingsFilePath = parser.value("settings");
  QString geometryFilePath = parser.value("geometry");

  // QString settingsFilePath = QObject::tr("/home/ubuntu/rafsine-gui/problems/data_center/settings.lua");
  // QString geometryFilePath = QObject::tr("/home/ubuntu/rafsine-gui/problems/data_center/geometry.lua");

  cudaProfilerStart();

  DomainData *domainData = new DomainData();
  SimulationWorker *simWorker = new SimulationWorker();
  if (!settingsFilePath.isEmpty() && !geometryFilePath.isEmpty())
  {
    domainData->loadFromLua(geometryFilePath.toUtf8().constData(), settingsFilePath.toUtf8().constData());
    simWorker->setDomainData(domainData);
  }

  MainWindow window(simWorker);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);

  const int retval = app.exec();

  cudaProfilerStop();
  cudaDeviceSynchronize();
  cudaDeviceReset();

  std::cout << "Exited with " << retval << std::endl;

  return retval;
}