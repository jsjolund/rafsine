#include "MainWindow.hpp"

MainWindow::MainWindow(SimulationThread *simThread)
    : m_simThread(simThread),
      m_widget(simThread, 1, 1, this),
      m_sliceMoveCounter(0)
{
  setCentralWidget(&m_widget);
  m_widget.setFocus();

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));
  m_secTimer->start(1000);

  m_msecTimer = new QTimer(this);
  connect(m_msecTimer, SIGNAL(timeout()), this, SLOT(msecUpdate()));
  m_msecTimer->start(50);

  m_statusLeft = new QLabel("No simulation loaded", this);
  // m_statusLeft->setFrameStyle(QFrame::Panel | QFrame::Plain);
  m_statusMiddle = new QLabel("", this);
  // m_statusMiddle->setFrameStyle(QFrame::Panel | QFrame::Plain);
  m_statusRight = new QLabel("", this);
  // m_statusRight->setFrameStyle(QFrame::Panel | QFrame::Plain);
  m_statusRight->setAlignment(Qt::AlignRight);
  statusBar()->addPermanentWidget(m_statusLeft, 1);
  statusBar()->addPermanentWidget(m_statusMiddle, 1);
  statusBar()->addPermanentWidget(m_statusRight, 1);

  createActions();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
  std::cout << "Exiting..." << std::endl;
}

void MainWindow::msecUpdate()
{
  m_widget.updateSlicePositions();
}

void MainWindow::secUpdate()
{
  if (m_simThread->hasDomainData())
  {
    SimulationTimer *simTimer = m_simThread->getDomainData()->m_simTimer;
    std::ostringstream stream;
    stream << "Time: " << *simTimer;
    stream << ", Rate: " << simTimer->getRealTimeRate();
    m_statusLeft->setText(QString::fromStdString(stream.str()));

    stream.str("");
    stream << "MLUPS: " << simTimer->getMLUPS();
    stream << ", LUPS: " << simTimer->getLUPS();
    m_statusRight->setText(QString::fromStdString(stream.str()));
  }
}

MainWindow::~MainWindow()
{
}

void MainWindow::open() { std::cout << "Open..." << std::endl; }
void MainWindow::rebuild() { std::cout << "Rebuilding voxel geometry" << std::endl; }
void MainWindow::resetFlow() { std::cout << "Resetting flow" << std::endl; }
void MainWindow::setOrthoCam() { std::cout << "Setting ortho " << camOrthoCheckBox->isChecked() << std::endl; }
void MainWindow::setShowLabels() { std::cout << "Setting labels " << showLabelsCheckBox->isChecked() << std::endl; }
void MainWindow::setDisplayModeSlice() { std::cout << "Setting slice" << std::endl; }
void MainWindow::setDisplayModeVoxel() { std::cout << "Setting vox" << std::endl; }
void MainWindow::setDisplayQuantityTemperature() { std::cout << "Quantity temperature" << std::endl; }
void MainWindow::setDisplayQuantityVelocity() { std::cout << "Quantity velocity" << std::endl; }
void MainWindow::setDisplayQuantityDensity() { std::cout << "Quantity density" << std::endl; }
void MainWindow::adjustDislayColors() { std::cout << "Adjust colors" << std::endl; }

void MainWindow::about()
{
  QString title = tr("About ").append(QCoreApplication::applicationName());
  QMessageBox::about(this, title,
                     tr("The <b>Application</b> example demonstrates how to "
                        "write modern GUI applications using Qt, with a menu bar, "
                        "toolbars, and a status bar."));
}

void MainWindow::createActions()
{
  //Simulation
  // 	Open*                        (Ctrl+O)
  // 	Rebuild*
  // 	Reset Flow*                  (F5)
  // 	--
  // 	Exit                         (Ctrl+Q)

  //Plot
  // 	Ortho camera
  // 	--
  // 	Slices/Voxels*                (F2/F3)
  // 	--
  // 	Density/Temperature/Velocity* (D/T/V)
  // 	--
  // 	Show labels
  // 	--
  // 	Adjust colors*                 (A)
  // 	Color scheme

  // Simulation menu
  QMenu *simMenu = menuBar()->addMenu(tr("&Simulation"));
  QToolBar *toolBar = addToolBar(tr("Simulation"));

  // Open file
  const QIcon openIcon = QIcon::fromTheme("document-open", QIcon(":assets/document-open.png"));
  QAction *openAct = new QAction(openIcon, tr("&Open Script..."), this);
  openAct->setStatusTip(tr("Open an existing LUA script file"));
  openAct->setShortcuts(QKeySequence::Open);
  connect(openAct, &QAction::triggered, this, &MainWindow::open);
  simMenu->addAction(openAct);
  toolBar->addAction(openAct);

  // Rebuild
  const QIcon rebuildIcon = QIcon::fromTheme("gtk-convert", QIcon(":assets/gtk-convert.png"));
  QAction *rebuildAct = new QAction(rebuildIcon, tr("Re&build"), this);
  rebuildAct->setStatusTip(tr("Rebuild from LUA script"));
  rebuildAct->setShortcut(Qt::Key_B | Qt::CTRL);
  connect(rebuildAct, &QAction::triggered, this, &MainWindow::rebuild);
  simMenu->addAction(rebuildAct);
  toolBar->addAction(rebuildAct);

  // Reset flow
  const QIcon resetIcon = QIcon::fromTheme("view-refresh", QIcon(":assets/view-refresh.png"));
  QAction *resetAct = new QAction(resetIcon, tr("&Reset Flow"), this);
  resetAct->setStatusTip(tr("Reset flow to initial conditions"));
  resetAct->setShortcut(Qt::Key_F5);
  connect(resetAct, &QAction::triggered, this, &MainWindow::resetFlow);
  simMenu->addAction(resetAct);
  toolBar->addAction(resetAct);

  simMenu->addSeparator();

  // Exit
  const QIcon exitIcon = QIcon::fromTheme("application-exit", QIcon(":assets/application-exit.png"));
  QAction *exitAct = simMenu->addAction(exitIcon, tr("E&xit"), this, &QWidget::close);
  exitAct->setShortcuts(QKeySequence::Quit);
  exitAct->setStatusTip(tr("Exit the application"));

  // Plot menu
  QMenu *plotMenu = menuBar()->addMenu(tr("&Plot"));

  // Slice/voxel mode
  QActionGroup *plotDisplayModeGroup = new QActionGroup(this);
  plotDisplayModeGroup->setExclusive(true);

  QAction *plotDisplayModeSlice = new QAction(tr("&Slices"), this);
  plotDisplayModeSlice->setStatusTip(tr("Display slices of simulation quantity"));
  plotDisplayModeSlice->setShortcut(Qt::Key_F2);
  plotDisplayModeGroup->addAction(plotDisplayModeSlice);
  plotDisplayModeSlice->setCheckable(true);
  plotDisplayModeSlice->setChecked(true);
  plotMenu->addAction(plotDisplayModeSlice);
  connect(plotDisplayModeSlice, &QAction::triggered, this, &MainWindow::setDisplayModeSlice);

  QAction *plotDisplayModeVoxel = new QAction(tr("Vo&xels"), this);
  plotDisplayModeSlice->setStatusTip(tr("Display voxel geometry"));
  plotDisplayModeVoxel->setShortcut(Qt::Key_F3);
  plotDisplayModeVoxel->setCheckable(true);
  plotDisplayModeGroup->addAction(plotDisplayModeVoxel);
  plotMenu->addAction(plotDisplayModeVoxel);
  connect(plotDisplayModeVoxel, &QAction::triggered, this, &MainWindow::setDisplayModeVoxel);

  plotMenu->addSeparator();

  // Temperature/Velocity/Density mode
  QActionGroup *plotDisplayQGroup = new QActionGroup(this);
  plotDisplayQGroup->setExclusive(true);

  const QIcon temperatureIcon = QIcon::fromTheme("temperature", QIcon(":assets/thermometer.png"));
  QAction *plotDisplayQTemp = new QAction(temperatureIcon, tr("&Temperature"), this);
  plotDisplayQTemp->setStatusTip(tr("Display slices show particle temperatures"));
  plotDisplayQGroup->addAction(plotDisplayQTemp);
  plotDisplayQTemp->setShortcut(Qt::Key_T);
  plotDisplayQTemp->setCheckable(true);
  plotDisplayQTemp->setChecked(true);
  plotMenu->addAction(plotDisplayQTemp);
  connect(plotDisplayQTemp, &QAction::triggered, this, &MainWindow::setDisplayQuantityTemperature);

  const QIcon velocityIcon = QIcon::fromTheme("velocity", QIcon(":assets/pinwheel.png"));
  QAction *plotDisplayQVel = new QAction(velocityIcon, tr("&Velocity"), this);
  plotDisplayQVel->setStatusTip(tr("Display slices show particle velocities"));
  plotDisplayQVel->setShortcut(Qt::Key_V);
  plotDisplayQVel->setCheckable(true);
  plotDisplayQGroup->addAction(plotDisplayQVel);
  plotMenu->addAction(plotDisplayQVel);
  connect(plotDisplayQVel, &QAction::triggered, this, &MainWindow::setDisplayQuantityVelocity);

  const QIcon densityIcon = QIcon::fromTheme("density", QIcon(":assets/density.png"));
  QAction *plotDisplayQDen = new QAction(densityIcon, tr("&Density"), this);
  plotDisplayQDen->setStatusTip(tr("Display slices show particle density"));
  plotDisplayQDen->setShortcut(Qt::Key_D);
  plotDisplayQDen->setCheckable(true);
  plotDisplayQGroup->addAction(plotDisplayQDen);
  plotMenu->addAction(plotDisplayQDen);
  connect(plotDisplayQDen, &QAction::triggered, this, &MainWindow::setDisplayQuantityDensity);

  plotMenu->addSeparator();

  const QIcon adjColorIcon = QIcon::fromTheme("image-adjust", QIcon(":assets/image-adjust.png"));
  QAction *adjColorAct = new QAction(adjColorIcon, tr("&Adjust colors"), this);
  adjColorAct->setStatusTip(tr("Adjust display slice colors to min/max"));
  adjColorAct->setShortcut(Qt::Key_A);
  connect(adjColorAct, &QAction::triggered, this, &MainWindow::adjustDislayColors);
  plotMenu->addAction(adjColorAct);

  // Show BC labels
  const QIcon showLabelsIcon = QIcon::fromTheme("insert-text", QIcon(":assets/insert-text.png"));
  showLabelsCheckBox = new QAction(showLabelsIcon, tr("&Show labels"), this);
  showLabelsCheckBox->setStatusTip(tr("Show boundary condition labels"));
  showLabelsCheckBox->setCheckable(true);
  showLabelsCheckBox->setChecked(true);
  connect(showLabelsCheckBox, &QAction::changed, this, &MainWindow::setShowLabels);
  plotMenu->addAction(showLabelsCheckBox);

  // Ortho cam
  camOrthoCheckBox = new QAction(tr("&Orthographic view"), this);
  camOrthoCheckBox->setStatusTip(tr("Use orthographic camera projection"));
  camOrthoCheckBox->setCheckable(true);
  connect(camOrthoCheckBox, &QAction::changed, this, &MainWindow::setOrthoCam);
  plotMenu->addAction(camOrthoCheckBox);

  plotMenu->addSeparator();

  // Help menu
  QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));
  QAction *aboutAct = helpMenu->addAction(tr("&About"), this, &MainWindow::about);
  aboutAct->setStatusTip(tr("Show the application's About box"));

  QAction *aboutQtAct = helpMenu->addAction(tr("About &Qt"), qApp, &QApplication::aboutQt);
  aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
}