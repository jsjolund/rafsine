#pragma once

#include <QAction>
#include <QActionGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDebug>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QGroupBox>
#include <QIcon>
#include <QKeySequence>
#include <QLabel>
#include <QList>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPointer>
#include <QRadioButton>
#include <QSplitter>
#include <QStatusBar>
#include <QString>
#include <QStyle>
#include <QThread>
#include <QTimer>
#include <QToolBar>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidgetAction>

#include <sstream>

#include "CFDTableView.hpp"
#include "CFDTreeWidget.hpp"
#include "CFDWidget.hpp"
#include "SimulationWorker.hpp"

#define LUA_SETTINGS_FILE_NAME "settings.lua"
#define LUA_GEOMETRY_FILE_NAME "geometry.lua"

/**
 * @brief Main window, containing all the GUI functionality
 * 
 */
class MainWindow : public QMainWindow {
  Q_OBJECT

 private:
  int m_sliceMoveCounter;

  // Shows a 3D visualization of the CFD simulation
  CFDWidget m_widget;
  // Tree widget showing details about boundary conditions
  CFDTreeWidget *m_tree;
  // Table for setting properties of dynamic boundary conditions
  CFDTableView *m_table;
  // Resizable splitters
  QSplitter *m_vSplitter, *m_hSplitter;
  // Pointer to the currently running simulation thread
  QThread *m_simThread;
  // Worker object for the simulation thread
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
  Q_SLOT void onTableEdited();

  explicit MainWindow(SimulationWorker *simWorker);
  virtual ~MainWindow();
  void closeEvent(QCloseEvent *event) override;
};
