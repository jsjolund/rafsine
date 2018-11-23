#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDesktopWidget>
#include <QObject>
#include <QThread>

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include "ConsoleClient.hpp"
#include "DomainData.hpp"
#include "MainWindow.hpp"
#include "SimulationWorker.hpp"

// #include "qt-unix-signals/sigwatch.h"
#include "sigwatch.h"

QCoreApplication *createApplication(int &argc, char *argv[]) {
  for (int i = 1; i < argc; ++i)
    if (!qstrcmp(argv[i], "-n") || !qstrcmp(argv[i], "--no-gui"))
      return new QCoreApplication(argc, argv);
  return new QApplication(argc, argv);
}

int main(int argc, char **argv) {
  Q_INIT_RESOURCE(res);
  QScopedPointer<QCoreApplication> appPtr(createApplication(argc, argv));

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
                                "Number of CUDA devices to use.", "8");
  QCommandLineOption headlessOpt({"n", "no-gui"}, "Run in headless mode");
  QCommandLineOption iterationsOpt(
      {"i", "iterations"},
      "Number of iterations to run before stopping the simulation.", "0");
  parser.addOption(settingsOpt);
  parser.addOption(geometryOpt);
  parser.addOption(devicesOpt);
  parser.addOption(headlessOpt);
  parser.addOption(iterationsOpt);
  parser.process(*appPtr);

  bool headless = parser.isSet(headlessOpt);
  QString settingsFilePath = parser.value("settings");
  QString geometryFilePath = parser.value("geometry");
  int iterations = parser.value("iterations").toInt();

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
  SimulationWorker *simWorker = new SimulationWorker(NULL, iterations);

  if (!settingsFilePath.isEmpty() && !geometryFilePath.isEmpty()) {
    domainData->loadFromLua(geometryFilePath.toUtf8().constData(),
                            settingsFilePath.toUtf8().constData());
    simWorker->setDomainData(domainData);
  }

  // Watch for unix signals
  UnixSignalWatcher sigwatch;
  sigwatch.watchForSignal(SIGINT);
  sigwatch.watchForSignal(SIGTERM);
  QObject::connect(&sigwatch, SIGNAL(unixSignal(int)),
                   (headless) ? qobject_cast<QCoreApplication *>(appPtr.data())
                              : qobject_cast<QApplication *>(appPtr.data()),
                   SLOT(quit()));

  int retval;
  if (headless) {
    QCoreApplication *app = qobject_cast<QCoreApplication *>(appPtr.data());
    ConsoleClient *client = new ConsoleClient(simWorker, numDevices, app);
    QObject::connect(app, SIGNAL(aboutToQuit()), client, SLOT(close()));
    QObject::connect(client, SIGNAL(finished()), app, SLOT(quit()));
    QTimer::singleShot(0, client, SLOT(run()));
    retval = app->exec();

  } else {
    QApplication *app = qobject_cast<QApplication *>(appPtr.data());
    MainWindow window(simWorker, numDevices);
    QObject::connect(app, SIGNAL(aboutToQuit()), &window, SLOT(close()));
    window.show();
    window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
    retval = app->exec();
  }
#pragma omp parallel num_threads(numDevices)
  {
    const int dev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaDeviceReset());
  }

  std::cout << "Exited with " << retval << std::endl;

  return retval;
}
