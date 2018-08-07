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

__global__ void kernel(uchar4 *ptr, float *colors, int dim)
{
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // now calculate the value at that position
  float fx = x / (float)dim - 0.5f;
  float fy = y / (float)dim - 0.5f;
  unsigned char intensity = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

  // accessing uchar4 vs unsigned char*
  ptr[offset].x = intensity * (1 - sin(colors[0])) / 2;
  ptr[offset].y = intensity * (1 - sin(colors[1])) / 2;
  ptr[offset].z = intensity * (1 - sin(colors[2])) / 2;
  ptr[offset].w = 255 * (1 - sin(colors[3])) / 2;
}

class CudaTestQuad : public CudaTexturedQuadGeometry
{
public:
  thrust::device_vector<float> *m_color_d;
  thrust::host_vector<float> *m_color_h;

  CudaTestQuad(unsigned int width, unsigned int height) : CudaTexturedQuadGeometry(width, height)
  {
    m_color_d = new thrust::device_vector<float>(4);
    m_color_h = new thrust::host_vector<float>(4);
    (*m_color_h)[0] = -osg::PI / 2;
    (*m_color_h)[1] = osg::PI / 2;
    (*m_color_h)[2] = 0;
    (*m_color_h)[3] = -osg::PI / 2;
  }

protected:
  virtual void runCudaKernel() const
  {
    dim3 grids(m_width / 16, m_height / 16);
    dim3 threads(16, 16);
    float *colorPtr = thrust::raw_pointer_cast(&(*m_color_d)[0]);
    uchar4 *devPtr = static_cast<uchar4 *>(m_texture->resourceData());
    kernel<<<grids, threads>>>(devPtr, colorPtr, m_width);
    cuda_check_errors("kernel");
    cudaDeviceSynchronize();
  }
};

class QtOSGWidget : public QOpenGLWidget
{
public:
  unsigned int m_imageWidth = 512;
  unsigned int m_imageHeight = 512;
  CudaTestQuad *m_quad;

  QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0)
      : QOpenGLWidget(parent), m_gfxWindow(new osgViewer::GraphicsWindowEmbedded(this->x(), this->y(),
                                                                                 this->width(), this->height())),
        m_viewer(new osgViewer::Viewer),
        m_scaleX(scaleX),
        m_scaleY(scaleY)
  {
    m_quad = new CudaTestQuad(m_imageWidth, m_imageHeight);
    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(m_quad);

    osg::Camera *camera = new osg::Camera;
    camera->setViewport(0, 0, this->width(), this->height());
    camera->setClearColor(osg::Vec4(0.9f, 0.9f, 0.9f, 1.f));
    float aspectRatio = static_cast<float>(this->width()) / static_cast<float>(this->height());
    camera->setProjectionMatrixAsPerspective(30.f, aspectRatio, 1.f, 1000.f);
    camera->setGraphicsContext(m_gfxWindow);

    m_viewer->setCamera(camera);
    m_viewer->setSceneData(geode);
    osgGA::TrackballManipulator *manipulator = new osgGA::TrackballManipulator;
    manipulator->setAllowThrow(false);
    this->setMouseTracking(true);
    m_viewer->setCameraManipulator(manipulator);
    m_viewer->addEventHandler(new osgViewer::StatsHandler);
    m_viewer->setRunFrameScheme(osgViewer::ViewerBase::FrameScheme::CONTINUOUS);
    m_viewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
    m_viewer->realize();
  }

  virtual ~QtOSGWidget() {}

  void setScale(qreal X, qreal Y)
  {
    m_scaleX = X;
    m_scaleY = Y;
    this->resizeGL(this->width(), this->height());
  }

protected:
  virtual void initializeGL()
  {
    osg::Geode *geode = dynamic_cast<osg::Geode *>(m_viewer->getSceneData());
    osg::StateSet *stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  }

  virtual void paintGL()
  {
    thrust::host_vector<float> *c = m_quad->m_color_h;
    const float d = 0.05f;
    (*c)[0] = (*c)[0] + d;
    (*c)[1] = (*c)[1] + d;
    (*c)[2] = (*c)[2] + d;

    *m_quad->m_color_d = *m_quad->m_color_h;

    m_viewer->frame();
  }

  virtual void resizeGL(int width, int height)
  {
    this->getEventQueue()->windowResize(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
    m_gfxWindow->resized(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
    osg::Camera *camera = m_viewer->getCamera();
    camera->setViewport(0, 0, this->width() * m_scaleX, this->height() * m_scaleY);
  }

  virtual void mouseMoveEvent(QMouseEvent *event)
  {
    this->getEventQueue()->mouseMotion(event->x() * m_scaleX, event->y() * m_scaleY);
  }

  virtual void mousePressEvent(QMouseEvent *event)
  {
    setFocus();
    unsigned int button = 0;
    switch (event->button())
    {
    case Qt::LeftButton:
      button = 1;
      break;
    case Qt::MiddleButton:
      button = 2;
      break;
    case Qt::RightButton:
      button = 3;
      break;
    default:
      break;
    }
    this->getEventQueue()->mouseButtonPress(event->x() * m_scaleX, event->y() * m_scaleY, button);
  }

  void keyPressEvent(QKeyEvent *event)
  {
    switch (event->key())
    {
    case Qt::Key_Escape:
      QApplication::quit();
      break;
    default:
      // Pass key to osgViewer::StatsHandler
      const char *keyData = event->text().toLatin1().data();
      m_gfxWindow->getEventQueue()->keyPress(osgGA::GUIEventAdapter::KeySymbol(*keyData));
      break;
    }
  }

  virtual void mouseReleaseEvent(QMouseEvent *event)
  {
    unsigned int button = 0;
    switch (event->button())
    {
    case Qt::LeftButton:
      button = 1;
      break;
    case Qt::MiddleButton:
      button = 2;
      break;
    case Qt::RightButton:
      button = 3;
      break;
    default:
      break;
    }
    this->getEventQueue()->mouseButtonRelease(event->x() * m_scaleX, event->y() * m_scaleY, button);
  }

  virtual void wheelEvent(QWheelEvent *event)
  {
    int delta = event->delta();
    osgGA::GUIEventAdapter::ScrollingMotion motion = delta > 0 ? osgGA::GUIEventAdapter::SCROLL_UP : osgGA::GUIEventAdapter::SCROLL_DOWN;
    this->getEventQueue()->mouseScroll(motion);
  }

  virtual bool event(QEvent *event)
  {
    bool handled = QOpenGLWidget::event(event);
    this->update();
    return handled;
  }

private:
  osgGA::EventQueue *getEventQueue() const
  {
    osgGA::EventQueue *eventQueue = m_gfxWindow->getEventQueue();
    return eventQueue;
  }

  osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_gfxWindow;
  osg::ref_ptr<osgViewer::Viewer> m_viewer;
  qreal m_scaleX, m_scaleY;
};

int main(int argc, char **argv)
{
  QApplication qapp(argc, argv);

  QMainWindow window;
  QtOSGWidget *widget = new QtOSGWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.3);
  widget->setFocus();

  return qapp.exec();
}