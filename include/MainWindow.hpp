#pragma once

#include <QAction>
#include <QActionGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDebug>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QIcon>
#include <QKeySequence>
#include <QLabel>
#include <QList>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QMutex>
#include <QPointer>
#include <QRadioButton>
#include <QSplitter>
#include <QStatusBar>
#include <QString>
#include <QStringList>
#include <QStyle>
#include <QTabWidget>
#include <QThread>
#include <QTimer>
#include <QToolBar>
#include <QTreeView>
#include <QVBoxLayout>
#include <QWidgetAction>

#include <sstream>

#include "AverageOutputTableView.hpp"
#include "BCInputTableView.hpp"
#include "CFDTreeWidget.hpp"
#include "CFDWidget.hpp"
#include "GitSHA1.hpp"
#include "LbmFile.hpp"
#include "SimulationWorker.hpp"

/**
 * @brief Main window, containing all the GUI functionality
 *
 */
class MainWindow : public QMainWindow {
  Q_OBJECT

 private:
  // Number of GPUs to use
  int m_numDevices;
  // True when the window is closing (to avoid
  // https://bugreports.qt.io/browse/QTBUG-43344)
  bool m_closing;
  // Currently opened lbm simulation file
  LbmFile m_lbmFile;
  // Shows a 3D visualization of the CFD simulation
  CFDWidget m_cfdWidget;
  // Tree widget showing details about boundary conditions
  CFDTreeWidget *m_tree;
  // Table for setting properties of dynamic boundary conditions
  BCInputTableView *m_inputTable;
  AverageOutputTableView *m_outputTable;
  QTabWidget *m_tabWidget;
  // Resizable splitters
  QSplitter *m_vSplitter, *m_hSplitter;
  // Pointer to the currently running simulation thread
  QThread *m_simThread;
  // Worker object for the simulation thread
  SimulationWorker *m_simWorker;
  QMutex m_mutex;

  QTimer *m_secTimer;
  QTimer *m_msecTimer;
  QLabel *m_statusLeft;
  QLabel *m_statusMiddle;
  QLabel *m_statusRight;

  Q_SLOT void secUpdate();

  QPointer<QAction> m_camOrthoCheckBox;
  QPointer<QAction> m_showLabelsCheckBox;
  QPointer<QAction> m_showSensorsCheckBox;
  QPointer<QAction> m_playPauseAction;

  void open();
  void rebuild();
  void resetFlow();
  void setOrthoCam();
  void setShowLabels();
  void setShowSensors();
  void setDisplayModeVoxel();
  void setDisplayModeSlice();
  void setDisplayModeDevices();
  void setDisplayQuantityTemperature();
  void setDisplayQuantityVelocity();
  void setDisplayQuantityDensity();
  void adjustDisplayColors();
  void setColorScheme(ColorScheme::Enum colorScheme);
  void pauseSimulation();
  void about();
  void hotkeys();
  void createActions();

  void loadSimulation(LbmFile lbmFile, int numDevices, uint64_t iterations);
  void destroySimulation();
  void closeSimulation();

 public:
  /**
   * @brief Slot triggered when editing the boundary condition table
   *
   * @return Q_SLOT onTableEdited
   */
  Q_SLOT void onTableEdited();
  /**
   * @brief Construct a new Main Window
   *
   * @param simWorker Simulation thread worker from program start
   */
  MainWindow(LbmFile lbmFile, int numDevices, uint64_t iterations);
  /**
   * @brief Destroy the Main Window
   *
   */
  virtual ~MainWindow();
  /**
   * @brief Callback for when user closed the Main Window
   *
   * @param event
   */
  void closeEvent(QCloseEvent *event) override;
};
