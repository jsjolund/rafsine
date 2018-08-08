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

class QtCFDWidget : public QtOSGWidget
{
public:
  unsigned int m_imageWidth = 512;
  unsigned int m_imageHeight = 512;
  osg::ref_ptr<SliceRender> m_sliceX;
  osg::ref_ptr<osg::Group> m_root;
  cudaStream_t m_renderStream;

  QtCFDWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0)
      : QtOSGWidget(scaleX, scaleY, parent)
  {
    m_sliceX = new SliceRender(m_imageWidth, m_imageHeight, m_renderStream);

    m_root = new osg::Group();
    m_root->addChild(m_sliceX->m_transform);
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

  // CUDA stream priorities. Simulation has highest priority, rendering lowest.
  cudaStream_t simStream;
  cudaStream_t renderStream;
  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
  cudaStreamCreateWithPriority(&simStream, cudaStreamNonBlocking, priority_high);
  cudaStreamCreateWithPriority(&renderStream, cudaStreamNonBlocking, priority_low);

  QApplication qapp(argc, argv);

  QMainWindow window;
  QtCFDWidget *widget = new QtCFDWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.3);
  widget->setFocus();
  widget->m_renderStream = renderStream;

  return qapp.exec();
}