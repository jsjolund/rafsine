#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDesktopWidget>
#include <QObject>
#include <QThread>

#include <stdio.h>
#include <unistd.h>
#include <iostream>

#include <sigwatch.h>

#include "ConsoleClient.hpp"
#include "DomainData.hpp"
#include "MainWindow.hpp"
#include "SimulationWorker.hpp"

static QCoreApplication *createApplication(int *argc, char *argv[]) {
  for (int i = 1; i < *argc; ++i)
    if (!qstrcmp(argv[i], "-n") || !qstrcmp(argv[i], "--no-gui"))
      return new QCoreApplication(*argc, argv);
  return new QApplication(*argc, argv);
}

static int getNumDevices(int numRequestedDevices) {
  int numDevices, numFoundDevices;
  CUDA_RT_CALL(cudaGetDeviceCount(&numFoundDevices));
  std::cout << "Found " << numFoundDevices << " CUDA GPU(s)" << std::endl;

  if (numRequestedDevices == 0) numRequestedDevices = numFoundDevices;
  if (numRequestedDevices <= numFoundDevices) {
    numDevices = numRequestedDevices;
  } else {
    std::cerr << "Invalid number of CUDA devices, only " << numFoundDevices
              << " available" << std::endl;
    return 0;
  }
  std::cout << "Using " << numDevices << " CUDA GPU(s)" << std::endl;
  return numDevices;
}

int main(int argc, char **argv) {
  Q_INIT_RESOURCE(res);
  QScopedPointer<QCoreApplication> appPtr(createApplication(&argc, argv));

  QCoreApplication::setOrganizationName("RISE SICS North");
  QCoreApplication::setApplicationName("LUA LBM GPU Leeds Luleå 2018");
  QCoreApplication::setApplicationVersion("v0.2");

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
  int numRequestedDevices = parser.value("devices").toInt();

  // Check that requested number of CUDA devices exist
  int numDevices = getNumDevices(numRequestedDevices);
  if (numDevices == 0) return 1;

  CUDA_RT_CALL(cudaProfilerStart());
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  // Load LUA scripts if provided
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
    // Use console client
    QCoreApplication *app = qobject_cast<QCoreApplication *>(appPtr.data());
    ConsoleClient *client = new ConsoleClient(simWorker, numDevices, app);
    QObject::connect(app, SIGNAL(aboutToQuit()), client, SLOT(close()));
    QObject::connect(client, SIGNAL(finished()), app, SLOT(quit()));
    QTimer::singleShot(0, client, SLOT(run()));
    retval = app->exec();

  } else {
    // Use QT client
    QApplication *app = qobject_cast<QApplication *>(appPtr.data());
    MainWindow window(simWorker, numDevices);
    QObject::connect(app, SIGNAL(aboutToQuit()), &window, SLOT(close()));
    window.show();
    window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
    retval = app->exec();
  }

  // Reset devices and exit
#pragma omp parallel num_threads(numDevices)
  {
    const int dev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaDeviceReset());
  }
  CUDA_RT_CALL(cudaProfilerStop());
  std::cout << "Exited with " << retval << std::endl;
  return retval;
}
