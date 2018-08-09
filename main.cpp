#include <QApplication>
#include <QMainWindow>
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QObject>
#include <QEvent>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QDesktopWidget>
#include <QScreen>
#include <QtGlobal>
#include <QWindow>

#include <osg/ref_ptr>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osg/Camera>
#include <osg/ShapeDrawable>
#include <osg/StateSet>
#include <osg/Material>
#include <osgGA/EventQueue>
#include <osgGA/TrackballManipulator>
#include <osg/StateAttribute>
#include <osg/Texture2D>
#include <osg/Vec3d>

#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIActionAdapter>
#include <osgGA/TrackballManipulator>

#include <osg/Image>
#include <osg/Texture>
#include <osgDB/ReadFile>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

#include "cuda/CudaTexturedQuadGeometry.hpp"
#include "cuda/CudaTextureSubloadCallback.hpp"
#include "cuda/CudaGraphicsResource.hpp"
#include "cuda/CudaTexture2D.hpp"
#include "cuda/CudaUtils.hpp"

#include "gui/QtOSGWidget.hpp"
#include "gui/SliceRender.hpp"

class QtCFDKeyboardHandler : public osgGA::GUIEventHandler
{
  bool handle(const osgGA::GUIEventAdapter &ea,
              osgGA::GUIActionAdapter &aa,
              osg::Object *, osg::NodeVisitor *)
  {
    typedef osgGA::GUIEventAdapter::KeySymbol osgKey;
    switch (ea.getEventType())
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):
      switch (ea.getKey())
      {
      case osgKey::KEY_Page_Down:
        std::cout << "pgdn" << std::endl;
        return true;
      case osgKey::KEY_Escape:
        QApplication::quit();
        return true;
      }
      break;
    }
    return false;
  }

  bool handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv)
  {
    return handle(*(event->asGUIEventAdapter()), *(nv->asEventVisitor()->getActionAdapter()), object, nv);
  }

  bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa)
  {
    return handle(ea, aa, NULL, NULL);
  }
};

class QtCFDWidget : public QtOSGWidget
{
public:
  unsigned int m_imageWidth = 256;
  unsigned int m_imageHeight = 256;
  osg::ref_ptr<SliceRender> m_sliceX, m_sliceY, m_sliceZ;
  osg::ref_ptr<osg::Group> m_root;
  osg::Vec3d m_slicePositions;

  QtCFDWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0)
      : QtOSGWidget(scaleX, scaleY, parent)
  {
    getViewer()->addEventHandler(new QtCFDKeyboardHandler());

    // CUDA stream priorities. Simulation has highest priority, rendering lowest.
    // This must be done in the thread which first runs a kernel?
    cudaStream_t simStream = 0;
    cudaStream_t renderStream = 0;
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&renderStream, cudaStreamDefault, priority_low);

    m_root = new osg::Group();

    float m = ((float)m_imageWidth) / 2;
    m_slicePositions.set(m, m, m);

    m_sliceX = new SliceRender(SliceRenderAxis::X_AXIS, m_imageWidth, m_imageHeight, renderStream);
    m_sliceX->getTransform()->setAttitude(osg::Quat(osg::PI / 2, osg::Vec3d(0, 0, 1)));
    m_sliceX->getTransform()->setPosition(osg::Vec3d(m_slicePositions.x(), 0, 0));
    m_root->addChild(m_sliceX->getTransform());

    m_sliceY = new SliceRender(SliceRenderAxis::Y_AXIS, m_imageWidth, m_imageHeight, renderStream);
    m_sliceY->getTransform()->setAttitude(osg::Quat(0, osg::Vec3d(0, 0, 1)));
    m_sliceY->getTransform()->setPosition(osg::Vec3d(0, m_slicePositions.y(), 0));
    m_root->addChild(m_sliceY->getTransform());

    m_sliceZ = new SliceRender(SliceRenderAxis::Z_AXIS, m_imageWidth, m_imageHeight, renderStream);
    m_sliceZ->getTransform()->setAttitude(osg::Quat(-osg::PI / 2, osg::Vec3d(1, 0, 0)));
    m_sliceZ->getTransform()->setPosition(osg::Vec3d(0, 0, m_slicePositions.z()));
    m_root->addChild(m_sliceZ->getTransform());

    getViewer()->setSceneData(m_root);
  }

  virtual void paintGL()
  {
    getViewer()->frame();
  }

  virtual void initializeGL()
  {
    for (unsigned int i = 0; i < m_root->getNumChildren(); i++)
    {
      osg::ref_ptr<osg::Node> childNode = m_root->getChild(i);
      osg::StateSet *stateSet = childNode->getOrCreateStateSet();
      stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    }
  }
};

int main(int argc, char **argv)
{

  QApplication qapp(argc, argv);

  QMainWindow window;
  QtCFDWidget *widget = new QtCFDWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.3);
  widget->setFocus();

  return qapp.exec();
}