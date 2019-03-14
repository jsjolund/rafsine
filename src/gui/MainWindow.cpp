#include "MainWindow.hpp"

MainWindow::MainWindow(LbmFile lbmFile, uint64_t iterations, int numDevices)
    : m_simWorker(NULL),
      m_numDevices(numDevices),
      m_widget(1, 1, this),
      m_closing(false) {
  m_hSplitter = new QSplitter(Qt::Horizontal, this);
  m_vSplitter = new QSplitter(Qt::Vertical, m_hSplitter);
  m_tree = new CFDTreeWidget(this);
  m_table = new CFDTableView(this);
  m_vSplitter->addWidget(m_tree);
  m_vSplitter->addWidget(m_table);
  m_hSplitter->addWidget(m_vSplitter);
  m_hSplitter->addWidget(&m_widget);
  m_hSplitter->show();
  m_hSplitter->setStretchFactor(0, 0);
  m_hSplitter->setStretchFactor(1, 1);
  setCentralWidget(m_hSplitter);
  m_widget.setFocus();

  m_secTimer = new QTimer(this);
  connect(m_secTimer, SIGNAL(timeout()), this, SLOT(secUpdate()));
  m_secTimer->start(1000);

  m_statusLeft = new QLabel("No simulation loaded", this);
  m_statusMiddle = new QLabel("", this);
  m_statusRight = new QLabel("", this);
  m_statusRight->setAlignment(Qt::AlignRight);
  statusBar()->addPermanentWidget(m_statusLeft, 1);
  statusBar()->addPermanentWidget(m_statusMiddle, 1);
  statusBar()->addPermanentWidget(m_statusRight, 1);

  createActions();

  m_simThread = new QThread;
  if (lbmFile.isValid()) {
    m_simWorker = new SimulationWorker(lbmFile, iterations, numDevices);
    m_simWorker->moveToThread(m_simThread);
    connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
    connect(m_simWorker, SIGNAL(finished()), m_simThread, SLOT(quit()));

    m_tree->buildModel(m_simWorker->getVoxelGeometry());
    m_table->buildModel(m_simWorker->getVoxelGeometry(),
                        m_simWorker->getUnitConverter());
    m_widget.setSimulationWorker(m_simWorker);
    m_simThread->start();
    m_widget.homeCamera();
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (!m_closing) {
    m_closing = true;
    if (m_simWorker) {
      m_simWorker->cancel();
      m_simThread->quit();
      std::cout << "Waiting for simulation threads..." << std::endl;
      m_simThread->wait();
    }
  }
  event->accept();
}

void MainWindow::onTableEdited() {
  if (m_simWorker) {
    std::vector<BoundaryCondition> *bcs = m_simWorker->getDomainData()->m_bcs;
    m_table->updateBoundaryConditions(bcs, m_simWorker->getVoxelGeometry(),
                                      m_simWorker->getUnitConverter());
    m_simWorker->uploadBCs();
  }
}

void MainWindow::secUpdate() {
  if (m_simWorker) {
    SimulationTimer *simTimer = m_simWorker->getDomainData()->m_timer;
    std::ostringstream stream;
    stream << "Time: " << *simTimer;
    stream << ", Rate: " << simTimer->getRealTimeRate();
    m_statusLeft->setText(QString::fromStdString(stream.str()));

    DomainData *params = m_simWorker->getDomainData();
    stream.str("");
    stream << "Lattice: (" << params->m_nx << ", " << params->m_ny << ", "
           << params->m_nz << ")";
    stream << ", MLUPS: " << simTimer->getMLUPS();
    stream << ", LUPS: " << simTimer->getLUPS();
    m_statusRight->setText(QString::fromStdString(stream.str()));
  }
}

MainWindow::~MainWindow() {}

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
    // QFile file(dlg.selectedFiles().at(0));
    // QFileInfo fileInfo(file);
    // LbmFile lbmFile(fileInfo.filePath());

    // m_statusLeft->setText(tr("Loading, please wait..."));
    // m_statusRight->setText(tr(""));
    // qApp->processEvents();

    // if (m_simThread->isRunning()) {
    //   m_simWorker->cancel();
    //   m_simThread->quit();
    //   std::cout << "Waiting for simulation threads..." << std::endl;
    //   m_simThread->wait();
    // }

    // delete m_simWorker;
    // m_simWorker = new SimulationWorker();
    // m_simWorker->moveToThread(m_simThread);
    // connect(m_simThread, SIGNAL(started()), m_simWorker, SLOT(run()));
    // connect(m_simWorker, SIGNAL(finished()), m_simThread, SLOT(quit()));

    // DomainData *domainData = new DomainData(m_numDevices);
    // domainData->loadFromLua(lbmFile.getGeometryPath(),
    //                         lbmFile.getSettingsPath());

    // m_simWorker->setDomainData(domainData);

    // m_tree->clear();
    // m_tree->buildModel(m_simWorker->getVoxelGeometry());
    // m_table->clear();
    // m_table->buildModel(m_simWorker->getVoxelGeometry(),
    //                     m_simWorker->getUnitConverter());

    // m_widget.getScene()->setVoxelGeometry(m_simWorker->getVoxelGeometry(),
    //                                       m_numDevices);

    // if (!m_simThread->isRunning()) {
    //   m_simWorker->resume();
    //   m_simThread->start();
    // }
    // m_widget.homeCamera();
  }
}

void MainWindow::setOrthoCam() {
  std::cout << "Setting ortho " << m_camOrthoCheckBox->isChecked() << std::endl;
}

void MainWindow::rebuild() {
  std::cout << "Rebuilding voxel geometry" << std::endl;
}

void MainWindow::resetFlow() {
  m_statusLeft->setText(tr("Resetting, please wait..."));
  m_statusRight->setText(tr(""));
  qApp->processEvents();
  if (m_simWorker) m_simWorker->resetDfs();
}

void MainWindow::setShowLabels() {
  m_widget.getScene()->setLabelsVisible(m_showLabelsCheckBox->isChecked());
}
void MainWindow::setDisplayModeSlice() {
  m_widget.getScene()->setDisplayMode(DisplayMode::SLICE);
}
void MainWindow::setDisplayModeVoxel() {
  m_widget.getScene()->setDisplayMode(DisplayMode::VOX_GEOMETRY);
}
void MainWindow::setDisplayModeDevices() {
  m_widget.getScene()->setDisplayMode(DisplayMode::DEVICES);
}
void MainWindow::setDisplayQuantityTemperature() {
  m_widget.getScene()->setDisplayQuantity(DisplayQuantity::TEMPERATURE);
}
void MainWindow::setDisplayQuantityVelocity() {
  m_widget.getScene()->setDisplayQuantity(DisplayQuantity::VELOCITY_NORM);
}
void MainWindow::setDisplayQuantityDensity() {
  m_widget.getScene()->setDisplayQuantity(DisplayQuantity::DENSITY);
}
void MainWindow::setColorScheme(ColorScheme::Enum colorScheme) {
  m_widget.getScene()->setColorScheme(colorScheme);
}
void MainWindow::adjustDisplayColors() { m_widget.adjustDisplayColors(); }

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
          .append("<tr><td>T</td><td>Show temperature</td></tr>")
          .append("<tr><td>V</td><td>Show velocity</td></tr>")
          .append("<tr><td>Insert</td><td>Slice X up</td></tr>")
          .append("<tr><td>Delete</td><td>Slice X down</td></tr>")
          .append("<tr><td>Home</td><td>Slice Y up</td></tr>")
          .append("<tr><td>End</td><td>Slice Y down</td></tr>")
          .append("<tr><td>Page Up</td><td>Slice Z up</td></tr>")
          .append("<tr><td>Page Down</td><td>Slice Z down</td></tr>")
          .append("<tr><td>Space</td><td>Pause simulation</td></tr>")
          .append("<tr><td>Esc</td><td>Exit</td></tr>")
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
  QMenu *simMenu = menuBar()->addMenu(tr("&Simulation"));
  QToolBar *toolBar = addToolBar(tr("Simulation"));

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
  QAction *openAct = new QAction(openIcon, tr("&Open Script..."), this);
  openAct->setStatusTip(tr("Open an existing LUA script file"));
  openAct->setShortcuts(QKeySequence::Open);
  connect(openAct, &QAction::triggered, this, &MainWindow::open);
  simMenu->addAction(openAct);
  toolBar->addAction(openAct);

  // Rebuild
  const QIcon rebuildIcon =
      QIcon::fromTheme("gtk-convert", QIcon(":assets/gtk-convert.png"));
  QAction *rebuildAct = new QAction(rebuildIcon, tr("Re&build"), this);
  rebuildAct->setStatusTip(tr("Rebuild from LUA script"));
  rebuildAct->setShortcut(Qt::Key_B | Qt::CTRL);
  connect(rebuildAct, &QAction::triggered, this, &MainWindow::rebuild);
  simMenu->addAction(rebuildAct);
  toolBar->addAction(rebuildAct);

  // Reset flow
  const QIcon resetIcon =
      QIcon::fromTheme("view-refresh", QIcon(":assets/view-refresh.png"));
  QAction *resetAct = new QAction(resetIcon, tr("&Reset Flow"), this);
  resetAct->setStatusTip(tr("Reset flow to initial conditions"));
  resetAct->setShortcut(Qt::Key_F5);
  connect(resetAct, &QAction::triggered, this, &MainWindow::resetFlow);
  simMenu->addAction(resetAct);
  toolBar->addAction(resetAct);

  simMenu->addSeparator();

  // Exit
  const QIcon exitIcon = QIcon::fromTheme(
      "application-exit", QIcon(":assets/application-exit.png"));
  QAction *exitAct =
      simMenu->addAction(exitIcon, tr("E&xit"), this, &QWidget::close);
  exitAct->setShortcuts(QKeySequence::Quit);
  exitAct->setStatusTip(tr("Exit the application"));

  // Plot menu
  QMenu *plotMenu = menuBar()->addMenu(tr("&Plot"));

  // Slice/voxel mode
  QActionGroup *plotDisplayModeGroup = new QActionGroup(this);
  plotDisplayModeGroup->setExclusive(true);

  QAction *plotDisplayModeSlice = new QAction(tr("&Slices"), this);
  plotDisplayModeSlice->setStatusTip(
      tr("Display slices of simulation quantity"));
  plotDisplayModeSlice->setShortcut(Qt::Key_F1);
  plotDisplayModeGroup->addAction(plotDisplayModeSlice);
  plotDisplayModeSlice->setCheckable(true);
  plotDisplayModeSlice->setChecked(true);
  plotMenu->addAction(plotDisplayModeSlice);
  connect(plotDisplayModeSlice, &QAction::triggered, this,
          &MainWindow::setDisplayModeSlice);

  QAction *plotDisplayModeVoxel = new QAction(tr("Vo&xels"), this);
  plotDisplayModeSlice->setStatusTip(tr("Display voxel geometry"));
  plotDisplayModeVoxel->setShortcut(Qt::Key_F2);
  plotDisplayModeVoxel->setCheckable(true);
  plotDisplayModeGroup->addAction(plotDisplayModeVoxel);
  plotMenu->addAction(plotDisplayModeVoxel);
  connect(plotDisplayModeVoxel, &QAction::triggered, this,
          &MainWindow::setDisplayModeVoxel);

  QAction *plotDisplayModeDevices =
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
  QActionGroup *plotDisplayQGroup = new QActionGroup(this);
  plotDisplayQGroup->setExclusive(true);

  const QIcon temperatureIcon =
      QIcon::fromTheme("temperature", QIcon(":assets/thermometer.png"));
  QAction *plotDisplayQTemp =
      new QAction(temperatureIcon, tr("&Temperature"), this);
  plotDisplayQTemp->setStatusTip(
      tr("Display slices show particle temperatures"));
  plotDisplayQGroup->addAction(plotDisplayQTemp);
  plotDisplayQTemp->setShortcut(Qt::Key_T);
  plotDisplayQTemp->setCheckable(true);
  plotDisplayQTemp->setChecked(true);
  plotMenu->addAction(plotDisplayQTemp);
  connect(plotDisplayQTemp, &QAction::triggered, this,
          &MainWindow::setDisplayQuantityTemperature);
  toolBar->addAction(plotDisplayQTemp);

  const QIcon velocityIcon =
      QIcon::fromTheme("velocity", QIcon(":assets/pinwheel.png"));
  QAction *plotDisplayQVel = new QAction(velocityIcon, tr("&Velocity"), this);
  plotDisplayQVel->setStatusTip(tr("Display slices show particle velocities"));
  plotDisplayQVel->setShortcut(Qt::Key_V);
  plotDisplayQVel->setCheckable(true);
  plotDisplayQGroup->addAction(plotDisplayQVel);
  plotMenu->addAction(plotDisplayQVel);
  connect(plotDisplayQVel, &QAction::triggered, this,
          &MainWindow::setDisplayQuantityVelocity);
  toolBar->addAction(plotDisplayQVel);

  const QIcon densityIcon =
      QIcon::fromTheme("density", QIcon(":assets/density.png"));
  QAction *plotDisplayQDen = new QAction(densityIcon, tr("&Density"), this);
  plotDisplayQDen->setStatusTip(tr("Display slices show particle density"));
  plotDisplayQDen->setShortcut(Qt::Key_D);
  plotDisplayQDen->setCheckable(true);
  plotDisplayQGroup->addAction(plotDisplayQDen);
  plotMenu->addAction(plotDisplayQDen);
  connect(plotDisplayQDen, &QAction::triggered, this,
          &MainWindow::setDisplayQuantityDensity);
  toolBar->addAction(plotDisplayQDen);

  plotMenu->addSeparator();

  // Show BC labels
  const QIcon showLabelsIcon =
      QIcon::fromTheme("insert-text", QIcon(":assets/insert-text.png"));
  m_showLabelsCheckBox = new QAction(showLabelsIcon, tr("&Show labels"), this);
  m_showLabelsCheckBox->setStatusTip(tr("Show boundary condition labels"));
  m_showLabelsCheckBox->setShortcut(Qt::Key_L);
  m_showLabelsCheckBox->setCheckable(true);
  m_showLabelsCheckBox->setChecked(true);
  connect(m_showLabelsCheckBox, &QAction::changed, this,
          &MainWindow::setShowLabels);
  plotMenu->addAction(m_showLabelsCheckBox);

  // Ortho cam
  m_camOrthoCheckBox = new QAction(tr("&Orthographic view"), this);
  m_camOrthoCheckBox->setStatusTip(tr("Use orthographic camera projection"));
  m_camOrthoCheckBox->setCheckable(true);
  connect(m_camOrthoCheckBox, &QAction::changed, this,
          &MainWindow::setOrthoCam);
  plotMenu->addAction(m_camOrthoCheckBox);

  plotMenu->addSeparator();
  toolBar->addSeparator();

  // Adjust slice colors min/max
  const QIcon adjColorIcon =
      QIcon::fromTheme("image-adjust", QIcon(":assets/image-adjust.png"));
  QAction *adjColorAct = new QAction(adjColorIcon, tr("&Adjust colors"), this);
  adjColorAct->setStatusTip(tr("Adjust display slice colors to min/max"));
  adjColorAct->setShortcut(Qt::Key_A);
  connect(adjColorAct, &QAction::triggered, this,
          &MainWindow::adjustDisplayColors);
  plotMenu->addAction(adjColorAct);
  toolBar->addAction(adjColorAct);

  // Set slice color scheme
  QMenu *colorSchemeMenu = plotMenu->addMenu(tr("&Color scheme"));

  QActionGroup *colorSchemeGroup = new QActionGroup(this);
  colorSchemeGroup->setExclusive(true);

  QAction *color0 = new QAction(tr("Black and White"), this);
  connect(color0, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::BLACK_AND_WHITE); });
  color0->setCheckable(true);
  color0->setShortcut(Qt::Key_1);
  colorSchemeGroup->addAction(color0);
  colorSchemeMenu->addAction(color0);

  QAction *color1 = new QAction(tr("Rainbow"), this);
  connect(color1, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::RAINBOW); });
  color1->setCheckable(true);
  color1->setShortcut(Qt::Key_2);
  colorSchemeGroup->addAction(color1);
  colorSchemeMenu->addAction(color1);

  QAction *color2 = new QAction(tr("Diverging"), this);
  connect(color2, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::DIVERGING); });
  color2->setCheckable(true);
  color2->setShortcut(Qt::Key_3);
  colorSchemeGroup->addAction(color2);
  colorSchemeMenu->addAction(color2);

  QAction *color3 = new QAction(tr("Oblivion"), this);
  connect(color3, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::OBLIVION); });
  color3->setCheckable(true);
  color3->setShortcut(Qt::Key_4);
  colorSchemeGroup->addAction(color3);
  colorSchemeMenu->addAction(color3);

  QAction *color4 = new QAction(tr("Blues"), this);
  connect(color4, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::BLUES); });
  color4->setCheckable(true);
  color4->setShortcut(Qt::Key_5);
  colorSchemeGroup->addAction(color4);
  colorSchemeMenu->addAction(color4);

  QAction *color5 = new QAction(tr("Sand"), this);
  connect(color5, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::SAND); });
  color5->setCheckable(true);
  color5->setShortcut(Qt::Key_6);
  colorSchemeGroup->addAction(color5);
  colorSchemeMenu->addAction(color5);

  QAction *color6 = new QAction(tr("Fire"), this);
  connect(color6, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::FIRE); });
  color6->setCheckable(true);
  color6->setShortcut(Qt::Key_7);
  colorSchemeGroup->addAction(color6);
  colorSchemeMenu->addAction(color6);

  QAction *color7 = new QAction(tr("ParaView"), this);
  connect(color7, &QAction::triggered, this,
          [this] { setColorScheme(ColorScheme::PARAVIEW); });
  color7->setCheckable(true);
  color7->setShortcut(Qt::Key_8);
  color7->setChecked(true);
  colorSchemeGroup->addAction(color7);
  colorSchemeMenu->addAction(color7);

  // Help menu
  QMenu *helpMenu = menuBar()->addMenu(tr("&Help"));

  QAction *hotkeyAct = helpMenu->addAction(tr("&Keyboard Shortcuts"), this,
                                           &MainWindow::hotkeys);
  hotkeyAct->setStatusTip(tr("Show available keyboard shortcuts"));

  QAction *aboutAct =
      helpMenu->addAction(tr("&About"), this, &MainWindow::about);
  aboutAct->setStatusTip(tr("Show the application's About box"));
}
