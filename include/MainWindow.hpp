
#ifndef __MAIN_WINDOW_GUARD_HPP
#define __MAIN_WINDOW_GUARD_HPP

#include <QMainWindow>
#include <QStatusBar>

#include "CFDWidget.hpp"
#include "SimulationThread.hpp"

class MainWindow : public QMainWindow
{
  Q_OBJECT

private:
  CFDWidget m_widget;
  SimulationThread *m_simThread;

public:
  MainWindow(SimulationThread *simThread);
  virtual ~MainWindow();
};

#endif /* end of include guard: __MAIN_WINDOW_GUARD_HPP */