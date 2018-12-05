
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

  /**
   * @brief Sets up the OpenGL resources and state. Gets called once before the
   * first time resizeGL() or paintGL() is called.
   */
  virtual void initializeGL() = 0;

  /**
   * @brief Called by paintGL when the scene is to be rendered.
   *
   * @param deltaTime Time (in ms) since last render
   */
  virtual void render(double deltaTime) = 0;

  /**
   * @brief Called by the parent class when the scene is to be rendered.
   *
   */
  virtual void paintGL();

  /**
   * @brief Called when the OpenGL widget needs to be resized.
   *
   * @param width
   * @param height
   */
  virtual void resizeGL(int width, int height);

  /**
   * @brief Reads a QMouseEvent and returns the OSG mouse button id
   *
   * @param event
   * @return unsigned int
   */
  unsigned int getMouseButton(QMouseEvent *event);

  /**
   * @brief Adds a mouse double click event to the OSG event queue.
   *
   * @param event
   */
  virtual void mouseDoubleClickEvent(QMouseEvent *event);

  /**
   * @brief Adds a mouse move event to the OSG event queue.
   *
   * @param event
   */
  virtual void mouseMoveEvent(QMouseEvent *event);

  /**
   * @brief Adds a mouse button press event to the OSG event queue.
   *
   * @param event
   */
  virtual void mousePressEvent(QMouseEvent *event);

  /**
   * @brief Adds a mouse button release event to the OSG event queue.
   *
   * @param event
   */
  virtual void mouseReleaseEvent(QMouseEvent *event);

  /**
   * @brief Adds a mouse wheel event to the OSG event queue.
   *
   * @param event
   */
  virtual void wheelEvent(QWheelEvent *event);

  /**
   * @brief Adds a keyboard button release event to the OSG event queue.
   *
   * @param event
   */
  void keyReleaseEvent(QKeyEvent *event);

  /**
   * @brief Adds a keyboard button press event to the OSG event queue.
   *
   * @param event
   */
  void keyPressEvent(QKeyEvent *event);

  /**
   * @brief Adds an event to the OSG event queue.
   *
   * @param event True if OSG handled the event
   */
  virtual bool event(QEvent *event);

 public:
  /**
   * @brief Construct a new QtOSGWidget
   *
   * @param scaleX TODO(Scaling factor)
   * @param scaleY
   * @param parent Parent QT widget
   */
  QtOSGWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0);
  virtual ~QtOSGWidget();

  /**
   * @brief Return the camera manipulator to default position
   *
   */
  void homeCamera();

  /**
   * @brief TODO()
   *
   * @param X
   * @param Y
   */
  void setScale(qreal X, qreal Y);
};
