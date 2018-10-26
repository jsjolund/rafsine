
#include "CFDWidget.hpp"

CFDWidget::CFDKeyboardHandler::CFDKeyboardHandler(CFDWidget *widget)
    : m_widget(widget), m_sliceXdir(0), m_sliceYdir(0), m_sliceZdir(0) {}

bool CFDWidget::CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea,
                                           osgGA::GUIActionAdapter &,
                                           osg::Object *, osg::NodeVisitor *) {
  typedef osgGA::GUIEventAdapter::KeySymbol osgKey;

  switch (ea.getEventType()) {
    case (osgGA::GUIEventAdapter::KEYDOWN):
      switch (ea.getKey()) {
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
      }
      break;
    case (osgGA::GUIEventAdapter::KEYUP):
      switch (ea.getKey()) {
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
      }
      break;
    default:
      return false;
  }
  return false;
}

bool CFDWidget::CFDKeyboardHandler::handle(osgGA::Event *event,
                                           osg::Object *object,
                                           osg::NodeVisitor *nv) {
  return handle(*(event->asGUIEventAdapter()),
                *(nv->asEventVisitor()->getActionAdapter()), object, nv);
}

bool CFDWidget::CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea,
                                           osgGA::GUIActionAdapter &aa) {
  return handle(ea, aa, NULL, NULL);
}

CFDWidget::CFDWidget(SimulationWorker *worker, qreal scaleX, qreal scaleY,
                     QWidget *parent)
    : QtOSGWidget(scaleX, scaleY, parent), m_simWorker(worker) {
  m_root = new osg::Group();

  m_scene = new CFDScene();
  m_root->addChild(m_scene->getRoot());

  if (m_simWorker->hasDomainData()) {
    m_scene->setVoxelGeometry(m_simWorker->getVoxelGeometry());
  }

  getViewer()->setSceneData(m_root);

  m_keyboardHandle = new CFDKeyboardHandler(this);
  getViewer()->addEventHandler(m_keyboardHandle);
  getViewer()->addEventHandler(new PickHandler(m_scene));

  getViewer()->getCamera()->setUpdateCallback(
      new CameraUpdateCallback(getViewer()->getCamera(), m_scene->getAxes()));

  m_root->addChild(m_scene->getHUDmatrix());
}

void CFDWidget::resizeGL(int width, int height) {
  m_scene->resize(width, height);
  QtOSGWidget::resizeGL(width, height);
}

void CFDWidget::updateSlicePositions() {
  if (m_simWorker->hasDomainData()) {
    m_scene->moveSlice(SliceRenderAxis::X_AXIS, m_keyboardHandle->m_sliceXdir);
    m_scene->moveSlice(SliceRenderAxis::Y_AXIS, m_keyboardHandle->m_sliceYdir);
    m_scene->moveSlice(SliceRenderAxis::Z_AXIS, m_keyboardHandle->m_sliceZdir);
  }
}

void CFDWidget::paintGL() {
  if (m_simWorker->hasDomainData()) {
    // Draw the CFD visualization slices
    m_simWorker->draw(m_scene->gpu_ptr(), m_scene->getDisplayQuantity());
  }
  // Draw the OSG widget
  getViewer()->frame();
}

void CFDWidget::initializeGL() {}
