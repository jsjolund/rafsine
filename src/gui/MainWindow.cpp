#include "MainWindow.hpp"

MainWindow::MainWindow(SimulationThread *simThread)
    : m_simThread(simThread),
      m_widget(simThread, 1, 1, this)
{
  setCentralWidget(&m_widget);
  m_widget.setFocus();
  statusBar()->showMessage(tr("Ready"));
}

MainWindow::~MainWindow()
{
}
