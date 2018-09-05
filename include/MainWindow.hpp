#pragma once

#include <QMainWindow>
#include <QStatusBar>
#include <QElapsedTimer>
#include <QTimer>
#include <QLabel>
#include <QIcon>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QString>
#include <QToolBar>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QMessageBox>
#include <QStyle>
#include <QRadioButton>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QWidgetAction>
#include <QPointer>
#include <QActionGroup>
#include <QKeySequence>

#include <sstream>

#include "CFDWidget.hpp"
#include "SimulationThread.hpp"

class MainWindow : public QMainWindow
{
  Q_OBJECT

protected:
  void closeEvent(QCloseEvent *event) override;

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

  QPointer<QAction> camOrthoCheckBox = nullptr;
  QPointer<QAction> showLabelsCheckBox = nullptr;

  void open();
  void rebuild();
  void resetFlow();
  void setOrthoCam();
  void setShowLabels();
  void setDisplayModeVoxel();
  void setDisplayModeSlice();
  void setDisplayQuantityTemperature();
  void setDisplayQuantityVelocity();
  void setDisplayQuantityDensity();
  void adjustDislayColors();
  
  void about();
  void createActions();

public:
  MainWindow(SimulationThread *simThread);
  virtual ~MainWindow();
};
