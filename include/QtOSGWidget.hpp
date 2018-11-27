
#pragma once

#include <QApplication>
#include <QMouseEvent>
#include <QOpenGLWidget>

#include <osg/ref_ptr>
#include <osgGA/EventQueue>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/OrbitManipulator>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include "CudaUtils.hpp"

/**
 * @brief Subclass of Camera Orbit manipulator, which disables some key press
 * captures.
 *
 */
class MyOrbitManipulator : public osgGA::OrbitManipulator {
  inline bool handleKeyDown(const osgGA::GUIEventAdapter &ea,
                            osgGA::GUIActionAdapter &us) {
    return false;
  }
};

/**
 * @brief Wrapper class which handles the communication between OSG and QT
 * OpenGL widgets.
 *
 */
class QtOSGWidget : public QOpenGLWidget {
 private:
  inline osgGA::EventQueue *getEventQueue() const {
    osgGA::EventQueue *eventQueue = m_gfxWindow->getEventQueue();
    return eventQueue;
  }
  double m_previousReferenceTime;

 protected:
  osg::ref_ptr<osgViewer::GraphicsWindowEmbedded>
      m_gfxWindow;  //!< Responsible for sending drawing calls to the QT widget
  osg::ref_ptr<osgViewer::Viewer>
      m_viewer;  //!< The embedded OSG graphics viewer
  osg::ref_ptr<osgViewer::StatsHandler>
      m_statsHandler;  //!< Gathers OpenGL statistics such as FPS
  osg::ref_ptr<osgGA::OrbitManipulator>
      m_cameraManipulator;  //!< Controls the camera with mouse input
  qreal m_scaleX;           //!< Horizontal scaling factor
  qreal m_scaleY;           //!< Vertical scaling factor

  virtual void initializeGL() = 0;
  virtual void render(double deltaTime) = 0;
  virtual void paintGL();

  virtual void resizeGL(int width, int height);

  unsigned int getMouseButton(QMouseEvent *event);
  virtual void mouseDoubleClickEvent(QMouseEvent *event);
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void mousePressEvent(QMouseEvent *event);
  virtual void mouseReleaseEvent(QMouseEvent *event);
  virtual void wheelEvent(QWheelEvent *event);

  void keyReleaseEvent(QKeyEvent *event);
  void keyPressEvent(QKeyEvent *event);

  virtual bool event(QEvent *event);

 public:
  QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0);
  virtual ~QtOSGWidget();

  void homeCamera();
  void setScale(qreal X, qreal Y);
};
