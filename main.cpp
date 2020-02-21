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
#include "LbmFile.hpp"
#include "MainWindow.hpp"
#include "SimulationWorker.hpp"

/**
 * @brief Create either a graphical or console environment depending on command
 * line arguments
 *
 * @param argc
 * @param argv
 * @return QCoreApplication*
 */
static QCoreApplication* createApplication(int* argc, char* argv[]) {
  for (int i = 1; i < *argc; ++i)
    if (!qstrcmp(argv[i], "-n") || !qstrcmp(argv[i], "--no-gui"))
      return new QCoreApplication(*argc, argv);
  return new QApplication(*argc, argv);
}

/**
 * @brief Main program entry point
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char** argv) {
  Q_INIT_RESOURCE(res);
  QScopedPointer<QCoreApplication> appPtr(createApplication(&argc, argv));

  QCoreApplication::setOrganizationName(ORGANIZATION_NAME);
  QCoreApplication::setApplicationName(APPLICATION_NAME);
  QCoreApplication::setApplicationVersion(APPLICATION_VERSION);

  // Register command line parser options
  QCommandLineParser parser;
  parser.setApplicationDescription(QCoreApplication::applicationName());
  parser.addHelpOption();
  parser.addVersionOption();
  QCommandLineOption lbmFileOpt({"f", "file"}, "LBM project file.",
                                "project.lbm");
  QCommandLineOption devicesOpt({"d", "devices"},
                                "Number of CUDA devices to use.", "1");
  QCommandLineOption headlessOpt({"n", "no-gui"}, "Run in headless mode");
  QCommandLineOption iterationsOpt(
      {"i", "iterations"},
      "Number of iterations to run before stopping the simulation.", "0");
  parser.addOption(lbmFileOpt);
  parser.addOption(devicesOpt);
  parser.addOption(headlessOpt);
  parser.addOption(iterationsOpt);
  parser.process(*appPtr);

  // Get the command line arguments
  bool headless = parser.isSet(headlessOpt);
  QString lbmFilePath = parser.value("file");
  int iterations = parser.value("iterations").toInt();

  // Number of CUDA devices
  int numRequestedDevices = parser.value("devices").toInt();
  int nd;
  CUDA_RT_CALL(cudaGetDeviceCount(&nd));
  if (numRequestedDevices > nd) {
    std::cerr << "Invalid number of CUDA devices" << numRequestedDevices
              << std::endl;
    return -1;
  } else if (numRequestedDevices > 0) {
    nd = numRequestedDevices;
  }
  std::cout << "Using " << nd << " CUDA GPU(s)" << std::endl;

  CUDA_RT_CALL(cudaProfilerStart());
  CUDA_RT_CALL(cudaSetDevice(0));
  CUDA_RT_CALL(cudaFree(0));

  LbmFile lbmFile;
  if (!lbmFilePath.isEmpty()) {
    lbmFile = LbmFile(lbmFilePath);
    if (!lbmFile.isValid()) {
      std::cerr << "Invalid LBM project file." << std::endl;
      return -1;
    }
  } else if (headless) {
    std::cerr << "No LBM project file specified." << std::endl;
    return -1;
  }

  // Watch for unix signals
  UnixSignalWatcher sigwatch;
  sigwatch.watchForSignal(SIGINT);
  sigwatch.watchForSignal(SIGTERM);
  QObject::connect(&sigwatch, SIGNAL(unixSignal(int)),
                   (headless) ? qobject_cast<QCoreApplication*>(appPtr.data())
                              : qobject_cast<QApplication*>(appPtr.data()),
                   SLOT(quit()));

  int retval;
  if (headless) {
    // Use console client
    QCoreApplication* app = qobject_cast<QCoreApplication*>(appPtr.data());
    ConsoleClient* client = new ConsoleClient(lbmFile, nd, iterations, app);
    QObject::connect(app, SIGNAL(aboutToQuit()), client, SLOT(close()));
    QObject::connect(client, SIGNAL(finished()), app, SLOT(quit()));
    QTimer::singleShot(0, client, SLOT(run()));
    retval = app->exec();

  } else {
    // Use QT client
    QApplication* app = qobject_cast<QApplication*>(appPtr.data());
    MainWindow window(lbmFile, nd);
    QObject::connect(app, SIGNAL(aboutToQuit()), &window, SLOT(close()));
    window.show();
    window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
    retval = app->exec();
  }

  // Reset devices and exit
#pragma omp parallel num_threads(nd)
  {
    const int dev = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CUDA_RT_CALL(cudaDeviceReset());
  }
  CUDA_RT_CALL(cudaProfilerStop());
  std::cout << "Exited with code: " << retval << std::endl;
  return retval;
}
