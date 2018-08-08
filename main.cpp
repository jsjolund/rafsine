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

class QtCFDWidget : public QtOSGWidget
{
public:
  unsigned int m_imageWidth = 512;
  unsigned int m_imageHeight = 512;
  CudaTestQuad *m_quad;

  QtCFDWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0)
      : QtOSGWidget(scaleX, scaleY, parent)
  {
    m_quad = new CudaTestQuad(m_imageWidth, m_imageHeight);
    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(m_quad);
    getViewer()->setSceneData(geode);
  }

  virtual void paintGL()
  {
    thrust::host_vector<float> *c = m_quad->m_color_h;
    const float d = 0.05f;
    (*c)[0] = (*c)[0] + d;
    (*c)[1] = (*c)[1] + d;
    (*c)[2] = (*c)[2] + d;

    *m_quad->m_color_d = *m_quad->m_color_h;

    getViewer()->frame();
  }

  virtual void initializeGL()
  {
    osg::Geode *geode = dynamic_cast<osg::Geode *>(getViewer()->getSceneData());
    osg::StateSet *stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
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