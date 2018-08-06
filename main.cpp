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

#include <osg/Image>
#include <osg/Texture>
#include <osgDB/ReadFile>

#include <iostream>
#include <unistd.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>

#include "gui/CudaGraphicsResource.hpp"
#include "gui/CudaTexture2D.hpp"

__global__ void kernel(uchar4 *ptr, int dim)
{
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // now calculate the value at that position
  float fx = x / (float)dim - 0.5f;
  float fy = y / (float)dim - 0.5f;
  unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

  // accessing uchar4 vs unsigned char*
  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

/// check if there is any error and display the details if there are some
inline void cuda_check_errors(const char *func_name)
{
  cudaError_t cerror = cudaGetLastError();
  if (cerror != cudaSuccess)
  {
    char host[256];
    gethostname(host, 256);
    printf("%s: CudaError: %s (on %s)\n", func_name, cudaGetErrorString(cerror), host);
    // exit(1);
  }
}

class MyQuad : public osg::Geometry
{
private:
  opencover::CudaTexture2D *texture;
  osg::Image *image1;
  unsigned int width, height;

public:
  float dt;
  MyQuad(unsigned int width, unsigned int height)
      : width(width),
        height(height),
        dt(1.0f),
        osg::Geometry(*osg::createTexturedQuadGeometry(
                          osg::Vec3(0.0f, 0.0f, 0.0f),
                          osg::Vec3(width, 0.0f, 0.0f),
                          osg::Vec3(0.0f, 0.0f, height),
                          0.0f,
                          0.0f,
                          1.0f,
                          1.0f),
                      osg::CopyOp::SHALLOW_COPY)
  {
    texture = new opencover::CudaTexture2D();
    osg::StateSet *stateset = getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
    stateset->setTextureAttribute(0, texture, osg::StateAttribute::ON);
    stateset->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::ON);
    texture->setDataVariance(osg::Object::DYNAMIC);
    setUseDisplayList(false);

    // image1 = new osg::Image();
    // image1->allocateImage(width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1);
    // texture->setImage(image1);
  }

  virtual void drawImplementation(osg::RenderInfo &renderInfo) const
  {
    osg::State *state = renderInfo.getState();
    // state->dirtyAllModes();

    if (texture->getTextureWidth() != width || texture->getTextureHeight() != height)
    {
      texture->setBorderWidth(0);
      texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
      texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
      texture->setTextureSize(width, height);
      texture->setSourceFormat(GL_RGBA);
      texture->setSourceType(GL_UNSIGNED_BYTE);
      texture->setInternalFormat(GL_RGBA8);
      texture->resize(state, width, height, 4);
    }

    dim3 grids(width / 16, height / 16);
    dim3 threads(16, 16);
    uchar4 *devPtr = static_cast<uchar4 *>(texture->resourceData());
    kernel<<<grids, threads>>>(devPtr, width);
    cuda_check_errors("kernel");
    // std::cout << abs(sin(dt)) << std::endl;
    // texture->dirty();
    // cudaDeviceSynchronize();

    osg::Geometry::drawImplementation(renderInfo);
    // texture->setImage(NULL);
  }
};

class QtOSGWidget : public QOpenGLWidget
{
public:
  unsigned int imageWidth = 512;
  unsigned int imageHeight = 512;

  thrust::device_vector<float> plot_d_;
  MyQuad *quad;

  float *plot_gpu_ptr()
  {
    return thrust::raw_pointer_cast(&(plot_d_)[0]);
  }

  QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0)
      : QOpenGLWidget(parent), _mGraphicsWindow(new osgViewer::GraphicsWindowEmbedded(this->x(), this->y(),
                                                                                      this->width(), this->height())),
        _mViewer(new osgViewer::Viewer),
        m_scaleX(scaleX),
        m_scaleY(scaleY),
        plot_d_(imageWidth * imageHeight)
  {
    quad = new MyQuad(imageWidth, imageWidth);
    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(quad);

    osg::Camera *camera = new osg::Camera;
    camera->setViewport(0, 0, this->width(), this->height());
    camera->setClearColor(osg::Vec4(0.9f, 0.9f, 0.9f, 1.f));
    float aspectRatio = static_cast<float>(this->width()) / static_cast<float>(this->height());
    camera->setProjectionMatrixAsPerspective(30.f, aspectRatio, 1.f, 1000.f);
    camera->setGraphicsContext(_mGraphicsWindow);

    _mViewer->setCamera(camera);
    _mViewer->setSceneData(geode);
    osgGA::TrackballManipulator *manipulator = new osgGA::TrackballManipulator;
    manipulator->setAllowThrow(false);
    this->setMouseTracking(true);
    _mViewer->setCameraManipulator(manipulator);
    _mViewer->addEventHandler(new osgViewer::StatsHandler);
    _mViewer->setThreadingModel(osgViewer::Viewer::SingleThreaded);
    _mViewer->realize();
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
    osg::Geode *geode = dynamic_cast<osg::Geode *>(_mViewer->getSceneData());
    osg::StateSet *stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  }

  virtual void paintGL()
  {
    quad->dt += 0.01f;
    _mViewer->frame();
  }

  virtual void resizeGL(int width, int height)
  {
    this->getEventQueue()->windowResize(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
    _mGraphicsWindow->resized(this->x() * m_scaleX, this->y() * m_scaleY, width * m_scaleX, height * m_scaleY);
    osg::Camera *camera = _mViewer->getCamera();
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
      exit(0);
      break;
    default:
      // Pass to osgViewer::StatsHandler
      const char *keyData = event->text().toLatin1().data();
      _mGraphicsWindow->getEventQueue()->keyPress(osgGA::GUIEventAdapter::KeySymbol(*keyData));
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
    osgGA::EventQueue *eventQueue = _mGraphicsWindow->getEventQueue();
    return eventQueue;
  }

  osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> _mGraphicsWindow;
  osg::ref_ptr<osgViewer::Viewer> _mViewer;
  qreal m_scaleX, m_scaleY;
};

int main(int argc, char **argv)
{
  QApplication qapp(argc, argv);

  QMainWindow window;
  QtOSGWidget *widget = new QtOSGWidget(1, 1, &window);
  window.setCentralWidget(widget);
  window.show();
  window.resize(QDesktopWidget().availableGeometry(&window).size() * 0.5);
  widget->setFocus();

  return qapp.exec();
}