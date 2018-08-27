
#include "CFDWidget.hpp"

CFDKeyboardHandler::CFDKeyboardHandler(CFDWidget *widget)
    : m_widget(widget) {}

bool CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea,
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
      m_widget->getScene()->moveSlice(SliceRenderAxis::Z_AXIS, -1);
      return true;
    case osgKey::KEY_Page_Up:
      m_widget->getScene()->moveSlice(SliceRenderAxis::Z_AXIS, 1);
      return true;
    case osgKey::KEY_End:
      m_widget->getScene()->moveSlice(SliceRenderAxis::Y_AXIS, -1);
      return true;
    case osgKey::KEY_Home:
      m_widget->getScene()->moveSlice(SliceRenderAxis::Y_AXIS, 1);
      return true;
    case osgKey::KEY_Delete:
      m_widget->getScene()->moveSlice(SliceRenderAxis::X_AXIS, -1);
      return true;
    case osgKey::KEY_Insert:
      m_widget->getScene()->moveSlice(SliceRenderAxis::X_AXIS, 1);
      return true;
    case osgKey::KEY_Escape:
      QApplication::quit();
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

bool CFDKeyboardHandler::handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv)
{
  return handle(*(event->asGUIEventAdapter()), *(nv->asEventVisitor()->getActionAdapter()), object, nv);
}

bool CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa)
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

  m_scene->setVoxelGeometry(m_simThread->getVoxelGeometry());

  getViewer()->setSceneData(m_root);

  getViewer()->addEventHandler(new PickHandler(m_scene));
  getViewer()->addEventHandler(new CFDKeyboardHandler(this));
}

void CFDWidget::paintGL()
{
  m_simThread->draw(m_scene->getPlot3d(), m_scene->getDisplayQuantity());
  cudaDeviceSynchronize();
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