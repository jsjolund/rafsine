
#include "CFDWidget.hpp"

CFDKeyboardHandler::CFDKeyboardHandler(CFDWidget *widget)
    : m_widget(widget) {}

bool CFDKeyboardHandler::handle(const osgGA::GUIEventAdapter &ea,
                                osgGA::GUIActionAdapter &aa,
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
      m_widget->m_scene->adjustDisplayColors();
    case osgKey::KEY_F1:
      m_widget->m_scene->setDisplayMode(DisplayMode::SLICE);
      return true;
    case osgKey::KEY_F2:
      m_widget->m_scene->setDisplayMode(DisplayMode::VOX_GEOMETRY);
      return true;
    case osgKey::KEY_Page_Down:
      m_widget->m_scene->moveSlice(SliceRenderAxis::Z_AXIS, -1);
      return true;
    case osgKey::KEY_Page_Up:
      m_widget->m_scene->moveSlice(SliceRenderAxis::Z_AXIS, 1);
      return true;
    case osgKey::KEY_End:
      m_widget->m_scene->moveSlice(SliceRenderAxis::Y_AXIS, -1);
      return true;
    case osgKey::KEY_Home:
      m_widget->m_scene->moveSlice(SliceRenderAxis::Y_AXIS, 1);
      return true;
    case osgKey::KEY_Delete:
      m_widget->m_scene->moveSlice(SliceRenderAxis::X_AXIS, -1);
      return true;
    case osgKey::KEY_Insert:
      m_widget->m_scene->moveSlice(SliceRenderAxis::X_AXIS, 1);
      return true;
    case osgKey::KEY_Escape:
      cudaStreamSynchronize(0);
      cudaDeviceSynchronize();
      cudaDeviceReset();
      QApplication::quit();
      return true;
    }
    break;
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

CFDWidget::CFDWidget(qreal scaleX, qreal scaleY, QWidget *parent)
    : QtOSGWidget(scaleX, scaleY, parent)
{

  getViewer()->addEventHandler(new CFDKeyboardHandler(this));

  m_root = new osg::Group();

  m_scene = new CFDScene();
  m_root->addChild(m_scene->getRoot());

  m_kernelData = new KernelData();
  osg::ref_ptr<VoxelMesh> mesh = new VoxelMesh(*(m_kernelData->vox->data));
  mesh->buildMesh(osg::Vec3i(-1, -1, -1), osg::Vec3i(-1, -1, -1));
  m_scene->setVoxelMesh(mesh);

  getViewer()->setSceneData(m_root);
}

void CFDWidget::paintGL()
{
  m_scene->frame(m_viewer->getCamera());
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