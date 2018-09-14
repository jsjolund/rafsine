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
#include <QFileDialog>
#include <QTreeView>
#include <QMessageLogger>
#include <QDebug>
#include <QThread>

#include <sstream>

#include "CFDWidget.hpp"
#include "SimulationWorker.hpp"

#define LUA_SETTINGS_FILE_NAME "settings.lua"
#define LUA_GEOMETRY_FILE_NAME "geometry.lua"

class MainWindow : public QMainWindow
{
  Q_OBJECT

protected:
  void closeEvent(QCloseEvent *event) override;

private:
  int m_sliceMoveCounter;

  CFDWidget m_widget;

  QThread *m_simThread;
  SimulationWorker *m_simWorker;

  QTimer *m_secTimer;
  QTimer *m_msecTimer;
  QLabel *m_statusLeft;
  QLabel *m_statusMiddle;
  QLabel *m_statusRight;

  Q_SLOT void secUpdate();
  Q_SLOT void msecUpdate();

  QPointer<QAction> m_camOrthoCheckBox;
  QPointer<QAction> m_showLabelsCheckBox;
  QPointer<QAction> m_playPauseAction;

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
  void adjustDisplayColors();
  void setColorScheme(ColorScheme::Enum colorScheme);
  void pauseSimulation();
  void about();
  void createActions();

public:
  MainWindow(SimulationWorker *simWorker);
  virtual ~MainWindow();
};
