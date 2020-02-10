#include "CFDWidget.hpp"

CFDWidget::CFDKeyboardHandler::CFDKeyboardHandler(CFDWidget* widget)
    : m_cfdWidget(widget), m_sliceXdir(0), m_sliceYdir(0), m_sliceZdir(0) {}

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

void CFDWidget::setSimulationWorker(SimulationWorker* simWorker,
                                    std::string voxMeshFilePath) {
  m_mutex.lock();
  m_simWorker = simWorker;
  if (m_simWorker) {
    int numDevices = m_simWorker->getNumDevices();
    m_scene->setVoxelGeometry(m_simWorker->getVoxels(), voxMeshFilePath,
                              numDevices);
  } else {
    m_scene->deleteVoxelGeometry();
  }
  m_mutex.unlock();
}

CFDWidget::CFDWidget(qreal scaleX, qreal scaleY, QWidget* parent)
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
  m_mutex.lock();
  if (m_simWorker) {
    real min, max;
    thrust::host_vector<real> histogram(HISTOGRAM_NUM_BINS);
    m_simWorker->getMinMax(&min, &max, &histogram);
    m_scene->adjustDisplayColors(min, max, histogram);
  }
  m_mutex.unlock();
}

void CFDWidget::resizeGL(int width, int height) {
  m_scene->resize(width, height);
  QtOSGWidget::resizeGL(width, height);
}

void CFDWidget::render(double deltaTime) {
  m_mutex.lock();
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
    if (m_scene->getDisplayMode() == DisplayMode::SLICE)
      m_simWorker->draw(m_scene->getDisplayQuantity(),
                        m_scene->getSlicePosition(), m_scene->getSliceX(),
                        m_scene->getSliceY(), m_scene->getSliceZ());
  }
  m_mutex.unlock();
}

void CFDWidget::initializeGL() {}
