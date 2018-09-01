
#ifndef __MAIN_WINDOW_GUARD_HPP
#define __MAIN_WINDOW_GUARD_HPP

#include <QMainWindow>
#include <QStatusBar>
#include <QElapsedTimer>
#include <QTimer>
#include <QLabel>

#include <sstream>

#include "CFDWidget.hpp"
#include "SimulationThread.hpp"

class MainWindow : public QMainWindow
{
  Q_OBJECT

private:
  int m_sliceMoveCounter;

  CFDWidget m_widget;
  SimulationThread *m_simThread;

  QTimer *m_secTimer;
  QTimer *m_msecTimer;
  QLabel *m_statusLeft;
  QLabel *m_statusMiddle;
  QLabel *m_statusRight;

  Q_SLOT void secUpdate();
  Q_SLOT void msecUpdate();

public:
  MainWindow(SimulationThread *simThread);
  virtual ~MainWindow();
};

#endif /* end of include guard: __MAIN_WINDOW_GUARD_HPP */