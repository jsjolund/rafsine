#include <QApplication>
#include <QMainWindow>
#include <QDesktopWidget>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include <cuda_profiler_api.h>

#include "CFDWidget.hpp"
#include "SimulationThread.hpp"

SimulationThread *thread;
cudaStream_t simStream = 0;
cudaStream_t renderStream = 0;

int main(int argc, char **argv)
{
  cudaProfilerStart();
  // CUDA stream priorities. Simulation has highest priority, rendering lowest.
  // This must be done in the thread which first runs a kernel
  int priorityHigh, priorityLow;
  cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
  cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priorityHigh);
  // cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, priorityLow);

  thread = new SimulationThread(new DomainData());
  thread->setSchedulePriority(OpenThreads::Thread::ThreadPriority ::THREAD_PRIORITY_MIN);

  QApplication app(argc, argv);

  QMainWindow window;
  CFDWidget *widget = new CFDWidget(thread, 1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
  widget->setFocus();

  thread->start();

  QObject::connect(&app, SIGNAL(aboutToQuit()), &window, SLOT(closing()));
  const int retval = app.exec();

  cudaProfilerStop();
  thread->cancel();
  thread->join();
  cudaStreamSynchronize(0);
  cudaDeviceSynchronize();
  cudaDeviceReset();

  return retval;
}