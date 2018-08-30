
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
  CFDWidget m_widget;
  SimulationThread *m_simThread;

  QTimer *m_timer;
  QLabel *m_statusLeft;
  QLabel *m_statusMiddle;
  QLabel *m_statusRight;

  Q_SLOT void update();

public:
  MainWindow(SimulationThread *simThread);
  virtual ~MainWindow();
};

#endif /* end of include guard: __MAIN_WINDOW_GUARD_HPP */