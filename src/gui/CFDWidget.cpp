
#include "CFDWidget.hpp"

CFDWidget::CFDKeyboardHandler::CFDKeyboardHandler(CFDWidget *widget)
    : m_widget(widget), m_sliceXdir(0), m_sliceYdir(0), m_sliceZdir(0) {}

bool CFDWidget::CFDKeyboardHandler::keyUp(int key) {
  typedef osgGA::GUIEventAdapter::KeySymbol osgKey;

  switch (key) {
    case osgKey::KEY_Page_Down:
      m_sliceZdir = 0;
      return true;
    case osgKey::KEY_Page_Up:
      m_sliceZdir = 0;
      return true;
    case osgKey::KEY_End:
      m_sliceYdir = 0;
      return true;
    case osgKey::KEY_Home:
      m_sliceYdir = 0;
      return true;
    case osgKey::KEY_Delete:
      m_sliceXdir = 0;
      return true;
    case osgKey::KEY_Insert:
      m_sliceXdir = 0;
      return true;
    default:
      return false;
  }
}

bool CFDWidget::CFDKeyboardHandler::keyDown(int key) {
  typedef osgGA::GUIEventAdapter::KeySymbol osgKey;

  switch (key) {
    case osgKey::KEY_Page_Down:
      m_sliceZdir = -1;
      return true;
    case osgKey::KEY_Page_Up:
      m_sliceZdir = 1;
      return true;
    case osgKey::KEY_End:
      m_sliceYdir = -1;
      return true;
    case osgKey::KEY_Home:
      m_sliceYdir = 1;
      return true;
    case osgKey::KEY_Delete:
      m_sliceXdir = -1;
      return true;
    case osgKey::KEY_Insert:
      m_sliceXdir = 1;
      return true;
    default:
      return false;
  }
}

void CFDWidget::setSimulationWorker(SimulationWorker *simWorker) {
  m_simWorker = simWorker;
  if (m_simWorker) {
    int numDevices = m_simWorker->getDomainData()->getNumDevices();
    m_scene->setVoxelGeometry(m_simWorker->getVoxelGeometry(), numDevices);
  }
}

CFDWidget::CFDWidget(qreal scaleX, qreal scaleY, QWidget *parent)
    : QtOSGWidget(scaleX, scaleY, parent),
      m_simWorker(NULL),
      m_sliceMoveCounter(0) {
  m_root = new osg::Group();

  m_scene = new CFDScene();
  m_root->addChild(m_scene);

  m_viewer->setSceneData(m_root);

  m_keyboardHandle = new CFDKeyboardHandler(this);
  m_viewer->addEventHandler(m_keyboardHandle);
  // m_viewer->addEventHandler(new PickHandler(m_scene));

  m_viewer->getCamera()->setUpdateCallback(
      new CameraUpdateCallback(m_viewer->getCamera(), m_scene->getAxes()));

  m_root->addChild(m_scene->getHUDmatrix());
}

void CFDWidget::adjustDisplayColors() {
  if (m_simWorker) {
    real min, max;
    m_simWorker->getMinMax(&min, &max);
    m_scene->adjustDisplayColors(min, max);
  }
}

void CFDWidget::resizeGL(int width, int height) {
  m_scene->resize(width, height);
  QtOSGWidget::resizeGL(width, height);
}

void CFDWidget::render(double deltaTime) {
  if (m_simWorker) {
    // Update slice positions if more than 50 ms passed
    m_sliceMoveCounter += deltaTime;
    if (m_sliceMoveCounter >= 0.05) {
      m_scene->moveSlice(D3Q4::X_AXIS, m_keyboardHandle->m_sliceXdir);
      m_scene->moveSlice(D3Q4::Y_AXIS, m_keyboardHandle->m_sliceYdir);
      m_scene->moveSlice(D3Q4::Z_AXIS, m_keyboardHandle->m_sliceZdir);
      m_sliceMoveCounter = 0;
    }

    // Draw the CFD visualization slices
    m_simWorker->draw(m_scene->getPlotArray(), m_scene->getDisplayQuantity(),
                      m_scene->getSlicePosition());
  }
  // Draw the OSG widget
  m_viewer->frame();
}

void CFDWidget::initializeGL() {}
