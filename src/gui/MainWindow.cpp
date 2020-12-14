#include "MainWindow.hpp"

MainWindow::MainWindow(LbmFile lbmFile, int nd)
    : m_nd(nd),
      m_closing(false),
      m_lbmFile(lbmFile),
      m_cfdWidget(1, 1, this),
      m_simWorker(NULL) {
  // Register signal types for table widgets
  qRegisterMetaType<QVector<int> >("QVector<int>");
  qRegisterMetaType<QList<QPersistentModelIndex> >(
      "QList<QPersistentModelIndex>");
  qRegisterMetaType<QAbstractItemModel::LayoutChangeHint>(
      "QAbstractItemModel::LayoutChangeHint");

  m_hSplitter = new QSplitter(Qt::Horizontal, this);
  m_vSplitter = new QSplitter(Qt::Vertical, m_hSplitter);
  m_tree = new CFDTreeWidget(this);
  m_inputTable = new BCInputTableView(this);
  m_outputTable = new AverageOutputTableView(this);
  m_tabWidget = new QTabWidget();
  m_tabWidget->addTab(m_vSplitter, tr("Inputs/Outputs"));

  m_vSplitter->addWidget(m_inputTable);
  m_vSplitter->addWidget(m_outputTable);
  m_tabWidget->addTab(m_tree, tr("Geometry"));

  m_hSplitter->addWidget(m_tabWidget);
  m_hSplitter->addWidget(&m_cfdWidget);
  m_hSplitter->show();
  m_hSplitter->setStretchFactor(0, 0);
  m_hSplitter->setStretchFactor(1, 1);
  setCentralWidget(m_hSplitter);
  m_cfdWidget.setFocus();

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));
  m_secTimer->start(1000);

  m_statusLeft = new QLabel("No simulation loaded", this);
  m_statusRight = new QLabel("", this);
  m_statusRight->setAlignment(Qt::AlignRight);
  statusBar()->addPermanentWidget(m_statusLeft, 1);
  statusBar()->addPermanentWidget(m_statusRight, 1);

  createActions();

  m_simThread = new QThread;
  if (lbmFile.isValid()) loadSimulation(lbmFile, nd);
}

void MainWindow::closeEvent(QCloseEvent* event) {
  if (!m_closing) {
    m_closing = true;
    m_mutex.lock();
    if (m_simWorker) {
      m_simWorker->cancel();
      m_simThread->quit();
      std::cout << "Waiting for simulation threads..." << std::endl;
      m_simThread->wait();
      std::cout << "Average MLUPS: "
                << m_simWorker->getSimulationTimer()->getAverageMLUPS()
                << std::endl;
    }
    m_mutex.unlock();
  }
  event->accept();
}

void MainWindow::onTableEdited() {
  m_mutex.lock();
  if (m_simWorker) {
    std::shared_ptr<BoundaryConditions> bcs =
        m_simWorker->getBoundaryConditions();
    m_inputTable->updateBoundaryConditions(bcs, m_simWorker->getVoxels(),
                                           m_simWorker->getUnitConverter());
    m_simWorker->uploadBCs();
  }
  m_mutex.unlock();
}

void MainWindow::secUpdate() {
  if (m_simWorker) {
    std::shared_ptr<SimulationTimer> simTimer =
        m_simWorker->getSimulationTimer();
    std::ostringstream stream;

    const time_t time =
        std::chrono::system_clock::to_time_t(simTimer->getTime());
    struct tm tm;
    localtime_r(&time, &tm);
    stream << "Time: " << std::put_time(&tm, "%F %T");

    // stream << "Time: " << *simTimer;
    stream << ", Rate: " << simTimer->getRealTimeRate();
    m_statusLeft->setText(QString::fromStdString(stream.str()));

    Vector3<size_t> size = m_simWorker->getDomainSize();
    stream.str("");
    stream << "Lattice: (" << size.x() << ", " << size.y() << ", " << size.z()
           << ")";
    stream << ", MLUPS: " << simTimer->getMLUPS();
    stream << ", LUPS: " << simTimer->getLUPS();
    m_statusRight->setText(QString::fromStdString(stream.str()));
  } else {
    m_statusLeft->setText(QString(tr("No simulation loaded")));
    m_statusRight->setText(QString());
  }
}

MainWindow::~MainWindow() {}

void MainWindow::destroySimulation() {
  m_cfdWidget.setSimulationWorker(NULL, std::string());
  m_inputTable->clear();
  m_outputTable->clear();
  m_tree->clear();
  if (m_simThread->isRunning()) {
    m_simWorker->cancel();
    m_simThread->quit();
    std::cout << "Waiting for simulation threads..." << std::endl;
    m_simThread->wait();
  }
  delete m_simWorker;
  m_simWorker = NULL;
}

void MainWindow::closeSimulation() {
  m_mutex.lock();
  destroySimulation();
  m_mutex.unlock();
  secUpdate();
  setWindowTitle(QCoreApplication::applicationName());
}

void MainWindow::loadSimulation(LbmFile lbmFile, int nd) {
  m_lbmFile = lbmFile;
  if (!m_lbmFile.isValid()) return;
  qApp->setOverrideCursor(Qt::WaitCursor);
  qApp->processEvents();

  m_simWorker = new SimulationWorker(lbmFile, nd);
  m_simWorker->moveToThread(m_simThread);
  connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
  connect(m_simWorker, SIGNAL(finished()), m_simThread, SLOT(quit()));

  m_tree->buildModel(m_simWorker->getVoxels());
  m_inputTable->buildModel(m_simWorker->getVoxels(),
                           m_simWorker->getUnitConverter());
  m_inputTable->setEditable(m_lbmFile.getInputCSVPath().length() == 0);

  m_outputTable->buildModel(*m_simWorker->getVoxels()->getSensors());
  m_simWorker->addAveragingObserver(m_outputTable);
  if (lbmFile.getOutputCSVPath().length() > 0)
    m_simWorker->addAveragingObserver(
        new CSVAveraging(lbmFile.getOutputCSVPath()));

  m_cfdWidget.setSimulationWorker(m_simWorker, m_simWorker->getVoxelMeshPath());
  std::cout << "Simulation '" << lbmFile.getTitle() << "' by '"
            << lbmFile.getAuthor() << "' successfully loaded" << std::endl;

  std::cout << "Starting simulation thread" << std::endl;
  m_simThread->start();
  m_cfdWidget.homeCamera();
  qApp->restoreOverrideCursor();

  setWindowTitle(QCoreApplication::applicationName().append(" - ").append(
      QString::fromUtf8(lbmFile.getTitle().c_str())));
}

void MainWindow::rebuild() {
  if (!m_lbmFile.isValid()) return;
  m_secTimer->stop();
  qApp->processEvents();

  m_mutex.lock();
  m_statusLeft->setText(tr("Rebuilding, please wait..."));
  m_statusRight->setText(tr(""));
  qApp->processEvents();
  destroySimulation();
  loadSimulation(m_lbmFile, m_nd);
  m_mutex.unlock();

  m_secTimer->start(1000);
}

void MainWindow::resetFlow() {
  if (!m_lbmFile.isValid()) return;
  m_secTimer->stop();
  qApp->processEvents();

  m_mutex.lock();
  m_statusLeft->setText(tr("Resetting, please wait..."));
  m_statusRight->setText(tr(""));
  qApp->processEvents();
  if (m_simWorker) m_simWorker->resetDfs();
  m_mutex.unlock();

  m_secTimer->start(1000);
}

void MainWindow::open() {
  QFileDialog dlg(nullptr, tr("Open LBM project file"));
  dlg.setOptions(QFileDialog::DontUseNativeDialog |
                 QFileDialog::DontResolveSymlinks);
  QStringList filters;
  filters << "LBM files (*.lbm)"
          << "Any files (*)";
  dlg.setNameFilters(filters);
  dlg.setFileMode(QFileDialog::ExistingFile);
  if (dlg.exec()) {
    QFile file(dlg.selectedFiles().at(0));
    QFileInfo fileInfo(file);
    LbmFile lbmFile(fileInfo.filePath());

    m_statusLeft->setText(tr("Loading, please wait..."));
    m_statusRight->setText(tr(""));
    qApp->processEvents();

    m_mutex.lock();
    destroySimulation();
    loadSimulation(lbmFile, m_nd);
    m_mutex.unlock();
  }
}

void MainWindow::setOrthoCam() {
  m_cfdWidget.setOrthographicCamera(m_camOrthoCheckBox->isChecked());
}
void MainWindow::setCameraX() { m_cfdWidget.setCameraAxis(D3Q4::Enum::X_AXIS); }
void MainWindow::setCameraY() { m_cfdWidget.setCameraAxis(D3Q4::Enum::Y_AXIS); }
void MainWindow::setCameraZ() { m_cfdWidget.setCameraAxis(D3Q4::Enum::Z_AXIS); }
void MainWindow::setShowLabels() {
  m_cfdWidget.getScene()->setLabelsVisible(m_showBCLabelsCheckBox->isChecked());
}
void MainWindow::setShowSensors() {
  m_cfdWidget.getScene()->setSensorsVisible(m_showSensorsCheckBox->isChecked());
}
void MainWindow::setDisplayModeSlice() {
  m_cfdWidget.getScene()->setDisplayMode(DisplayMode::SLICE);
}
void MainWindow::setDisplayModeVoxel() {
  m_cfdWidget.getScene()->setDisplayMode(DisplayMode::VOX_GEOMETRY);
}
void MainWindow::setDisplayModeDevices() {
  m_cfdWidget.getScene()->setDisplayMode(DisplayMode::DEVICES);
}
void MainWindow::setDisplayQuantityTemperature() {
  m_cfdWidget.getScene()->setDisplayQuantity(DisplayQuantity::TEMPERATURE);
}
void MainWindow::setDisplayQuantityVelocity() {
  m_cfdWidget.getScene()->setDisplayQuantity(DisplayQuantity::VELOCITY_NORM);
}
void MainWindow::setDisplayQuantityDensity() {
  m_cfdWidget.getScene()->setDisplayQuantity(DisplayQuantity::DENSITY);
}
void MainWindow::setColorScheme(ColorScheme::Enum colorScheme) {
  m_cfdWidget.getScene()->setColorScheme(colorScheme);
}
void MainWindow::adjustDisplayColors() { m_cfdWidget.adjustDisplayColors(); }

void MainWindow::about() {
  QString title = QString().append(QCoreApplication::applicationName());
  QString version =
      tr("Version: ").append(QCoreApplication::applicationVersion());
  // Git commit tag is automatically updated by CMake
  QString commit = tr("Commit: ").append(g_GIT_SHA1);
  QString text = tr("<b>")
                     .append(title)
                     .append("</b>")
                     .append("<br>")
                     .append(version)
                     .append("<br>")
                     .append(commit);
  QMessageBox::about(this, title, text);
}

void MainWindow::hotkeys() {
  QString title = QString().append(QCoreApplication::applicationName());
  QString text =
      tr("<table cellspacing=1 cellpadding = 2 style='border-width: 1px; ")
          .append("border-style: groove; border-color: #000000'>")
          .append("<tr><th>Keyboard</th><th>Action</th></tr>")
          .append("<tr><td>F1</td><td>Display slices</td></tr>")
          .append("<tr><td>F2</td><td>Display geometry</td></tr>")
          .append("<tr><td>F3</td><td>Display domain decomposition</td></tr>")
          .append("<tr><td>F5</td><td>Restart simulation</td></tr>")
          .append("<tr><td>1-8</td><td>Set color scheme</td></tr>")
          .append("<tr><td>A</td><td>Adjust slice colors min/max</td></tr>")
          .append("<tr><td>D</td><td>Show density</td></tr>")
          .append("<tr><td>L</td><td>Show boundary condition labels</td></tr>")
          .append("<tr><td>R</td><td>Show averaging labels</td></tr>")
          .append("<tr><td>S</td><td>Show 3D performance statistics</td></tr>")
          .append("<tr><td>T</td><td>Show temperature</td></tr>")
          .append("<tr><td>V</td><td>Show velocity</td></tr>")
          .append("<tr><td>Insert</td><td>Slice X up</td></tr>")
          .append("<tr><td>Delete</td><td>Slice X down</td></tr>")
          .append("<tr><td>Home</td><td>Slice Y up</td></tr>")
          .append("<tr><td>End</td><td>Slice Y down</td></tr>")
          .append("<tr><td>Page Up</td><td>Slice Z up</td></tr>")
          .append("<tr><td>Page Down</td><td>Slice Z down</td></tr>")
          .append("<tr><td>Space</td><td>Pause simulation</td></tr>")
          .append("<tr><th>Mouse</th><th>Action</th>")
          .append("<tr><td>Left click+drag</td><td>Rotate camera</td></tr>")
          .append("<tr><td>Middle click+drag</td><td>Pan camera</td></tr>")
          .append("<tr><td>Right click+drag</td><td>Zoom in/out</td></tr>")
          .append("<tr><td>Scroll wheel</td><td>Zoom in/out</td></tr>")
          .append("</table>");
  QMessageBox::about(this, title, text);
}

void MainWindow::pauseSimulation() {
  if (m_simWorker) {
    if (m_simThread->isRunning()) {
      m_simWorker->cancel();
      m_simThread->quit();
      m_simThread->wait();
      const QIcon startIcon = QIcon::fromTheme(
          "media-playback-start", QIcon(":assets/media-playback-start.png"));
      m_playPauseAction->setIcon(startIcon);
    } else {
      m_simWorker->resume();
      m_simThread->start();
      const QIcon pauseIcon = QIcon::fromTheme(
          "media-playback-pause", QIcon(":assets/media-playback-pause.png"));
      m_playPauseAction->setIcon(pauseIcon);
    }
  }
}

void MainWindow::createActions() {
  // Simulation menu
  QMenu* simMenu = menuBar()->addMenu(tr("&Simulation"));
  QToolBar* toolBar = addToolBar(tr("Simulation"));

  // Play/pause
  const QIcon pauseIcon = QIcon::fromTheme(
      "media-playback-pause", QIcon(":assets/media-playback-pause.png"));
  m_playPauseAction = new QAction(pauseIcon, tr("&Pause simulation"), this);
  m_playPauseAction->setStatusTip(tr("Pause/Resume the simulation"));
  m_playPauseAction->setShortcut(Qt::Key_Space);
  connect(m_playPauseAction, &QAction::triggered, this,
          &MainWindow::pauseSimulation);
  toolBar->addAction(m_playPauseAction);

  // Open file
  const QIcon openIcon =
      QIcon::fromTheme("document-open", QIcon(":assets/document-open.png"));
  QAction* openAct = new QAction(openIcon, tr("&Open File..."), this);
  openAct->setStatusTip(tr("Open an existing LBM file"));
  openAct->setShortcuts(QKeySequence::Open);
  connect(openAct, &QAction::triggered, this, &MainWindow::open);
  simMenu->addAction(openAct);
  toolBar->addAction(openAct);

  // Rebuild
  const QIcon rebuildIcon =
      QIcon::fromTheme("gtk-convert", QIcon(":assets/gtk-convert.png"));
  QAction* rebuildAct = new QAction(rebuildIcon, tr("Re&build"), this);
  rebuildAct->setStatusTip(tr("Rebuild"));
  rebuildAct->setShortcut(Qt::Key_B | Qt::CTRL);
  connect(rebuildAct, &QAction::triggered, this, &MainWindow::rebuild);
  simMenu->addAction(rebuildAct);
  toolBar->addAction(rebuildAct);

  // Reset flow
  const QIcon resetIcon =
      QIcon::fromTheme("view-refresh", QIcon(":assets/view-refresh.png"));
  QAction* resetAct = new QAction(resetIcon, tr("&Reset Flow"), this);
  resetAct->setStatusTip(tr("Reset flow to initial conditions"));
  resetAct->setShortcut(Qt::Key_F5);
  connect(resetAct, &QAction::triggered, this, &MainWindow::resetFlow);
  simMenu->addAction(resetAct);
  toolBar->addAction(resetAct);

  // Close file
  const QIcon closeIcon =
      QIcon::fromTheme("edit-delete", QIcon(":assets/edit-delete.png"));
  QAction* closeAct = new QAction(closeIcon, tr("&Close File"), this);
  closeAct->setStatusTip(tr("Close current LBM file"));
  closeAct->setShortcuts(QKeySequence::Close);
  connect(closeAct, &QAction::triggered, this, &MainWindow::closeSimulation);
  simMenu->addAction(closeAct);
  toolBar->addAction(closeAct);

  simMenu->addSeparator();

  // Exit
  const QIcon exitIcon = QIcon::fromTheme(
      "application-exit", QIcon(":assets/application-exit.png"));
  QAction* exitAct =
      simMenu->addAction(exitIcon, tr("E&xit"), this, &QWidget::close);
  exitAct->setShortcuts(QKeySequence::Quit);
  exitAct->setStatusTip(tr("Exit the application"));

  // Plot menu
  QMenu* plotMenu = menuBar()->addMenu(tr("&Plot"));

  // Slice/voxel mode
  QActionGroup* plotDisplayModeGroup = new QActionGroup(this);
  plotDisplayModeGroup->setExclusive(true);

  QAction* plotDisplayModeSlice = new QAction(tr("&Slices"), this);
  plotDisplayModeSlice->setStatusTip(
      tr("Display slices of simulation quantity"));
  plotDisplayModeSlice->setShortcut(Qt::Key_F1);
  plotDisplayModeGroup->addAction(plotDisplayModeSlice);
  plotDisplayModeSlice->setCheckable(true);
  plotDisplayModeSlice->setChecked(true);
  plotMenu->addAction(plotDisplayModeSlice);
  connect(plotDisplayModeSlice, &QAction::triggered, this,
          &MainWindow::setDisplayModeSlice);

  QAction* plotDisplayModeVoxel = new QAction(tr("Vo&xels"), this);
  plotDisplayModeSlice->setStatusTip(tr("Display voxel geometry"));
  plotDisplayModeVoxel->setShortcut(Qt::Key_F2);
  plotDisplayModeVoxel->setCheckable(true);
  plotDisplayModeGroup->addAction(plotDisplayModeVoxel);
  plotMenu->addAction(plotDisplayModeVoxel);
  connect(plotDisplayModeVoxel, &QAction::triggered, this,
          &MainWindow::setDisplayModeVoxel);

  QAction* plotDisplayModeDevices =
      new QAction(tr("&Domain Decomposition"), this);
  plotDisplayModeSlice->setStatusTip(
      tr("Display CUDA device domain decomposition"));
  plotDisplayModeDevices->setShortcut(Qt::Key_F3);
  plotDisplayModeDevices->setCheckable(true);
  plotDisplayModeGroup->addAction(plotDisplayModeDevices);
  plotMenu->addAction(plotDisplayModeDevices);
  connect(plotDisplayModeDevices, &QAction::triggered, this,
          &MainWindow::setDisplayModeDevices);

  plotMenu->addSeparator();
  toolBar->addSeparator();

  // Temperature/Velocity/Density mode
  QActionGroup* plotDisplayQGroup = new QActionGroup(this);
  plotDisplayQGroup->setExclusive(true);

  const QIcon temperatureIcon =
      QIcon::fromTheme("temperature", QIcon(":assets/thermometer.png"));
  QAction* plotTemperature =
      new QAction(temperatureIcon, tr("&Temperature"), this);
  plotTemperature->setStatusTip(
      tr("Display slices show particle temperatures"));
  plotDisplayQGroup->addAction(plotTemperature);
  plotTemperature->setShortcut(Qt::Key_T);
  plotTemperature->setCheckable(true);
  plotTemperature->setChecked(true);
  plotMenu->addAction(plotTemperature);
  connect(plotTemperature, &QAction::triggered, this,
          &MainWindow::setDisplayQuantityTemperature);
  toolBar->addAction(plotTemperature);

  const QIcon velocityIcon =
      QIcon::fromTheme("velocity", QIcon(":assets/pinwheel.png"));
  QAction* plotVelocity = new QAction(velocityIcon, tr("&Velocity"), this);
  plotVelocity->setStatusTip(tr("Display slices show particle velocities"));
  plotVelocity->setShortcut(Qt::Key_V);
  plotVelocity->setCheckable(true);
  plotDisplayQGroup->addAction(plotVelocity);
  plotMenu->addAction(plotVelocity);
  connect(plotVelocity, &QAction::triggered, this,
          &MainWindow::setDisplayQuantityVelocity);
  toolBar->addAction(plotVelocity);

  const QIcon densityIcon =
      QIcon::fromTheme("density", QIcon(":assets/density.png"));
  QAction* plotDensity = new QAction(densityIcon, tr("&Density"), this);
  plotDensity->setStatusTip(tr("Display slices show particle density"));
  plotDensity->setShortcut(Qt::Key_D);
  plotDensity->setCheckable(true);
  plotDisplayQGroup->addAction(plotDensity);
  plotMenu->addAction(plotDensity);
  connect(plotDensity, &QAction::triggered, this,
          &MainWindow::setDisplayQuantityDensity);
  toolBar->addAction(plotDensity);

  plotMenu->addSeparator();

  // Show BC labels
  const QIcon showLabelsIcon =
      QIcon::fromTheme("insert-text", QIcon(":assets/insert-text.png"));
  m_showBCLabelsCheckBox =
      new QAction(showLabelsIcon, tr("&Show boundary condition labels"), this);
  m_showBCLabelsCheckBox->setStatusTip(tr("Show boundary condition labels"));
  m_showBCLabelsCheckBox->setShortcut(Qt::Key_L);
  m_showBCLabelsCheckBox->setCheckable(true);
  m_showBCLabelsCheckBox->setChecked(false);
  connect(m_showBCLabelsCheckBox, &QAction::changed, this,
          &MainWindow::setShowLabels);
  plotMenu->addAction(m_showBCLabelsCheckBox);

  // Show sensors
  const QIcon showSensorsIcon =
      QIcon::fromTheme("measure", QIcon(":assets/measure.png"));
  m_showSensorsCheckBox =
      new QAction(showSensorsIcon, tr("&Show averaging labels"), this);
  m_showSensorsCheckBox->setStatusTip(tr("Show averaging labels"));
  m_showSensorsCheckBox->setShortcut(Qt::Key_R);
  m_showSensorsCheckBox->setCheckable(true);
  m_showSensorsCheckBox->setChecked(false);
  connect(m_showSensorsCheckBox, &QAction::changed, this,
          &MainWindow::setShowSensors);
  plotMenu->addAction(m_showSensorsCheckBox);

  // Ortho cam
  m_camOrthoCheckBox = new QAction(tr("&Orthographic view"), this);
  m_camOrthoCheckBox->setStatusTip(tr("Use orthographic camera projection"));
  m_camOrthoCheckBox->setCheckable(true);
  m_camOrthoCheckBox->setShortcut(Qt::Key_O);
  connect(m_camOrthoCheckBox, &QAction::changed, this,
          &MainWindow::setOrthoCam);
  plotMenu->addAction(m_camOrthoCheckBox);

  plotMenu->addSeparator();
  toolBar->addSeparator();

  // Adjust slice colors min/max
  const QIcon adjColorIcon =
      QIcon::fromTheme("image-adjust", QIcon(":assets/image-adjust.png"));
  QAction* adjColorAct = new QAction(adjColorIcon, tr("&Adjust colors"), this);
  adjColorAct->setStatusTip(tr("Adjust display slice colors to min/max"));
  adjColorAct->setShortcut(Qt::Key_A);
  connect(adjColorAct, &QAction::triggered, this,
          &MainWindow::adjustDisplayColors);
  toolBar->addAction(adjColorAct);

  // Camera view along x-axis
  const QIcon camXIcon =
      QIcon::fromTheme("image-adjust", QIcon(":assets/xaxis.png"));
  QAction* camXAct = new QAction(camXIcon, tr("&Camera view X-axis"), this);
  camXAct->setStatusTip(tr("Set the camera view along X-axis"));
  camXAct->setShortcut(Qt::Key_X);
  connect(camXAct, &QAction::triggered, this, &MainWindow::setCameraX);
  plotMenu->addAction(camXAct);
  toolBar->addAction(camXAct);

  // Camera view along y-axis
  const QIcon camYIcon =
      QIcon::fromTheme("image-adjust", QIcon(":assets/yaxis.png"));
  QAction* camYAct = new QAction(camYIcon, tr("&Camera view Y-axis"), this);
  camYAct->setStatusTip(tr("Set the camera view along Y-axis"));
  camYAct->setShortcut(Qt::Key_Y);
  connect(camYAct, &QAction::triggered, this, &MainWindow::setCameraY);
  plotMenu->addAction(camYAct);
  toolBar->addAction(camYAct);

  // Camera view along z-axis
  const QIcon camZIcon =
      QIcon::fromTheme("image-adjust", QIcon(":assets/zaxis.png"));
  QAction* camZAct = new QAction(camXIcon, tr("&Camera view Z-axis"), this);
  camZAct->setStatusTip(tr("Set the camera view along Z-axis"));
  camZAct->setShortcut(Qt::Key_Z);
  connect(camZAct, &QAction::triggered, this, &MainWindow::setCameraZ);
  plotMenu->addAction(camZAct);
  toolBar->addAction(camZAct);

  plotMenu->addSeparator();
  plotMenu->addAction(adjColorAct);

  // Set slice color scheme
  QMenu* colorSchemeMenu = plotMenu->addMenu(tr("&Color scheme"));

  QActionGroup* colorSchemeGroup = new QActionGroup(this);
  colorSchemeGroup->setExclusive(true);

  QAction* color0 = new QAction(tr("Black and White"), this);
  connect(color0, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::BLACK_AND_WHITE); });
  color0->setCheckable(true);
  color0->setShortcut(Qt::Key_1);
  colorSchemeGroup->addAction(color0);
  colorSchemeMenu->addAction(color0);

  QAction* color1 = new QAction(tr("Rainbow"), this);
  connect(color1, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::RAINBOW); });
  color1->setCheckable(true);
  color1->setShortcut(Qt::Key_2);
  colorSchemeGroup->addAction(color1);
  colorSchemeMenu->addAction(color1);

  QAction* color2 = new QAction(tr("Diverging"), this);
  connect(color2, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::DIVERGING); });
  color2->setCheckable(true);
  color2->setShortcut(Qt::Key_3);
  colorSchemeGroup->addAction(color2);
  colorSchemeMenu->addAction(color2);

  QAction* color3 = new QAction(tr("Oblivion"), this);
  connect(color3, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::OBLIVION); });
  color3->setCheckable(true);
  color3->setShortcut(Qt::Key_4);
  colorSchemeGroup->addAction(color3);
  colorSchemeMenu->addAction(color3);

  QAction* color4 = new QAction(tr("Blues"), this);
  connect(color4, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::BLUES); });
  color4->setCheckable(true);
  color4->setShortcut(Qt::Key_5);
  colorSchemeGroup->addAction(color4);
  colorSchemeMenu->addAction(color4);

  QAction* color5 = new QAction(tr("Sand"), this);
  connect(color5, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::SAND); });
  color5->setCheckable(true);
  color5->setShortcut(Qt::Key_6);
  colorSchemeGroup->addAction(color5);
  colorSchemeMenu->addAction(color5);

  QAction* color6 = new QAction(tr("Fire"), this);
  connect(color6, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::FIRE); });
  color6->setCheckable(true);
  color6->setShortcut(Qt::Key_7);
  colorSchemeGroup->addAction(color6);
  colorSchemeMenu->addAction(color6);

  QAction* color7 = new QAction(tr("ParaView"), this);
  connect(color7, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::PARAVIEW); });
  color7->setCheckable(true);
  color7->setShortcut(Qt::Key_8);
  color7->setChecked(true);
  colorSchemeGroup->addAction(color7);
  colorSchemeMenu->addAction(color7);

  // Help menu
  QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));

  QAction* hotkeyAct = helpMenu->addAction(tr("&Keyboard Shortcuts"), this,
                                           &MainWindow::hotkeys);
  hotkeyAct->setStatusTip(tr("Show available keyboard shortcuts"));

  QAction* aboutAct =
      helpMenu->addAction(tr("&About"), this, &MainWindow::about);
  aboutAct->setStatusTip(tr("Show the application's About box"));
}
