
#include "CFDWidget.hpp"

CFDWidget::CFDKeyboardHandler::CFDKeyboardHandler(CFDWidget *widget)
    : m_widget(widget),
      m_sliceXdir(0),
      m_sliceYdir(0),
      m_sliceZdir(0)
{
}

bool CFDWidget::CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea,
                                           osgGA::GUIActionAdapter &,
                                           osg::Object *,
                                           osg::NodeVisitor *)
{
  typedef osgGA::GUIEventAdapter::KeySymbol osgKey;

  switch (ea.getEventType())
  {
  case (osgGA::GUIEventAdapter::KEYDOWN):
    switch (ea.getKey())
    {
    case ' ':
      m_widget->m_simThread->pause(!m_widget->m_simThread->isPaused());
      return true;
    case 'a':
      m_widget->getScene()->adjustDisplayColors();
      return true;
    case 't':
      m_widget->getScene()->setDisplayQuantity(DisplayQuantity::TEMPERATURE);
      return true;
    case 'v':
      m_widget->getScene()->setDisplayQuantity(DisplayQuantity::VELOCITY_NORM);
      return true;
    case 'd':
      m_widget->getScene()->setDisplayQuantity(DisplayQuantity::DENSITY);
      return true;
    case osgKey::KEY_F1:
      m_widget->getScene()->setDisplayMode(DisplayMode::SLICE);
      return true;
    case osgKey::KEY_F2:
      m_widget->getScene()->setDisplayMode(DisplayMode::VOX_GEOMETRY);
      return true;
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
    case osgKey::KEY_Escape:
      QApplication::quit();
      return true;
    }
    break;
  case (osgGA::GUIEventAdapter::KEYUP):
    switch (ea.getKey())
    {
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
  case (osgGA::GUIEventAdapter::CLOSE_WINDOW):
    QApplication::quit();
    return true;
  case (osgGA::GUIEventAdapter::QUIT_APPLICATION):
    QApplication::quit();
    return true;
  default:
    return false;
  }
  return false;
}

bool CFDWidget::CFDKeyboardHandler::handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv)
{
  return handle(*(event->asGUIEventAdapter()), *(nv->asEventVisitor()->getActionAdapter()), object, nv);
}

bool CFDWidget::CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa)
{
  return handle(ea, aa, NULL, NULL);
}

CFDWidget::CFDWidget(SimulationThread *thread,
                     qreal scaleX,
                     qreal scaleY,
                     QWidget *parent)
    : QtOSGWidget(scaleX, scaleY, parent), m_simThread(thread)
{
  m_root = new osg::Group();

  m_scene = new CFDScene();
  m_root->addChild(m_scene->getRoot());

  if (m_simThread->hasDomainData())
  {
    m_scene->setVoxelGeometry(m_simThread->getVoxelGeometry());
    getViewer()->addEventHandler(new PickHandler(m_scene));
  }

  getViewer()->setSceneData(m_root);

  m_keyboardHandle = new CFDKeyboardHandler(this);
  getViewer()->addEventHandler(m_keyboardHandle);
  m_timer.setStartTick();
  m_lastTime = m_timer.getStartTick();
}

void CFDWidget::updateSlicePositions()
{
  m_scene->moveSlice(SliceRenderAxis::X_AXIS, m_keyboardHandle->m_sliceXdir);
  m_scene->moveSlice(SliceRenderAxis::Y_AXIS, m_keyboardHandle->m_sliceYdir);
  m_scene->moveSlice(SliceRenderAxis::Z_AXIS, m_keyboardHandle->m_sliceZdir);
}

void CFDWidget::paintGL()
{
  if (m_simThread->hasDomainData())
  {
    m_simThread->draw(m_scene->getPlot3d(), m_scene->getDisplayQuantity());
    cudaDeviceSynchronize();
  }
  getViewer()->frame();
}

void CFDWidget::initializeGL()
{
  for (unsigned int i = 0; i < m_root->getNumChildren(); i++)
  {
    osg::ref_ptr<osg::Node> childNode = m_root->getChild(i);
    osg::StateSet *stateSet = childNode->getOrCreateStateSet();
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  }
}