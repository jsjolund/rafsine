#include "MainWindow.hpp"

MainWindow::MainWindow(SimulationThread *simThread)
    : m_simThread(simThread),
      m_widget(simThread, 1, 1, this)
{
  setCentralWidget(&m_widget);
  m_widget.setFocus();

  m_timer = new QTimer(this);
  connect(m_timer, SIGNAL(timeout()), this, SLOT(update()));
  m_timer->start(1000);

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

  // QMetaObject::connectSlotsByName(this);
}

void MainWindow::update()
{
  if (m_simThread->hasDomainData())
  {
    SimulationTimer *simTimer = m_simThread->getDomainData()->m_simTimer;
    std::ostringstream stream;
    stream << "Time: " << *simTimer;
    stream << " Rate: " << simTimer->getRealTimeRate();
    m_statusLeft->setText(QString::fromStdString(stream.str()));

    stream.str("");
    stream << "MLUPS: " << simTimer->getMLUPS();
    stream << " LUPS: " << simTimer->getLUPS();
    m_statusRight->setText(QString::fromStdString(stream.str()));
  }
}

MainWindow::~MainWindow()
{
}
