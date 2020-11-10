
#pragma once

#include <QApplication>
#include <QMouseEvent>
#include <QOpenGLWidget>
#include <algorithm>
#include <osg/Timer>
#include <osg/ref_ptr>
#include <osgGA/EventQueue>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/OrbitManipulator>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include "CudaUtils.hpp"

#define BACKGROUND_COLOR osg::Vec4(0.0, 0.0, 0.0, 1.0)

/**
 * @brief Subclass of OrbitManipulator, to handle orthographic camera mode
 *
 */
class MyOrbitManipulator : public osgGA::OrbitManipulator {
 private:
  osg::ref_ptr<osg::Camera> m_camera;
  bool m_orthoCamera;

 protected:
  /**
   * @brief Prevent camera home when space is pressed
   *
   * @return true
   * @return false
   */
  inline bool handleKeyDown(const osgGA::GUIEventAdapter&,
                            osgGA::GUIActionAdapter&) {
    return false;
  }

  virtual void zoomModel(const float dy, bool pushForwardIfNeeded = true) {
    if (m_orthoCamera) {
      double left, right, bottom, top, zNear, zFar;
      m_camera->getProjectionMatrixAsOrtho(left, right, bottom, top, zNear,
                                           zFar);
      double scl = -100;
      double aspect = bottom / left;
      double dw = dy * scl;
      double dh = dy * aspect * scl;
      left = fminf(left + dw, -1.0);
      right = fmaxf(right - dw, 1.0);
      bottom = fminf(bottom + dh, -aspect);
      top = fmaxf(top - dh, aspect);
      m_camera->setProjectionMatrixAsOrtho(left, right, bottom, top, zNear,
                                           zFar);
    } else {
      osgGA::OrbitManipulator::zoomModel(dy, pushForwardIfNeeded);
    }
  }

 public:
  /**
   * @brief Set camera to orthographic mode
   *
   * @param state
   * @param w
   * @param h
   */
  void setOrthographicCamera(bool state, float w, float h) {
    m_orthoCamera = state;
    if (state) {
      m_camera->setProjectionMatrixAsOrtho(-w / 2, w / 2, -h / 2, h / 2, -1.0,
                                           1.0);
    } else {
      m_camera->setProjectionMatrixAsPerspective(30.0, w / h, 1.0, 1000.0);
    }
  }

  /**
   * @brief Check if camera is in orthographic mode
   *
   * @return true
   * @return false
   */
  bool isOrthographicCamera() { return m_orthoCamera; }

  explicit MyOrbitManipulator(osg::ref_ptr<osg::Camera> camera)
      : osgGA::OrbitManipulator(), m_camera(camera), m_orthoCamera(false) {}
};

/**
 * @brief Wrapper class which handles the communication between OSG and QT
 * OpenGL widgets.
 *
 */
class QtOSGWidget : public QOpenGLWidget {
 private:
  inline osgGA::EventQueue* getEventQueue() const {
    osgGA::EventQueue* eventQueue = m_gfxWindow->getEventQueue();
    return eventQueue;
  }
  double m_prevRefTime;

 protected:
  //! Responsible for sending drawing calls to the QT widget
  osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_gfxWindow;
  //! The embedded OSG graphics viewer
  osg::ref_ptr<osgViewer::Viewer> m_viewer;
  //! Gathers OpenGL statistics such as FPS
  osg::ref_ptr<osgViewer::StatsHandler> m_statsHandler;
  //! Controls the camera with mouse input
  osg::ref_ptr<MyOrbitManipulator> m_cameraManipulator;
  //! Horizontal scaling factor
  qreal m_scaleX;
  //! Vertical scaling factor
  qreal m_scaleY;

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
  unsigned int getMouseButton(QMouseEvent* event);

  /**
   * @brief Adds a mouse double click event to the OSG event queue.
   *
   * @param event
   */
  virtual void mouseDoubleClickEvent(QMouseEvent* event);

  /**
   * @brief Adds a mouse move event to the OSG event queue.
   *
   * @param event
   */
  virtual void mouseMoveEvent(QMouseEvent* event);

  /**
   * @brief Adds a mouse button press event to the OSG event queue.
   *
   * @param event
   */
  virtual void mousePressEvent(QMouseEvent* event);

  /**
   * @brief Adds a mouse button release event to the OSG event queue.
   *
   * @param event
   */
  virtual void mouseReleaseEvent(QMouseEvent* event);

  /**
   * @brief Adds a mouse wheel event to the OSG event queue.
   *
   * @param event
   */
  virtual void wheelEvent(QWheelEvent* event);

  /**
   * @brief Adds a keyboard button release event to the OSG event queue.
   *
   * @param event
   */
  void keyReleaseEvent(QKeyEvent* event);

  /**
   * @brief Adds a keyboard button press event to the OSG event queue.
   *
   * @param event
   */
  void keyPressEvent(QKeyEvent* event);

  /**
   * @brief Adds an event to the OSG event queue.
   *
   * @param event True if OSG handled the event
   */
  virtual bool event(QEvent* event);

 public:
  /**
   * @brief Construct a new QtOSGWidget
   *
   * @param scaleX TODO(Scaling factor)
   * @param scaleY
   * @param parent Parent QT widget
   */
  QtOSGWidget(qreal scaleX, qreal scaleY, QWidget* parent = 0);
  virtual ~QtOSGWidget();

  /**
   * @brief Return the camera manipulator to default position
   *
   */
  void homeCamera();
  /**
   * @brief Toggle orthographic camera projection on/off
   *
   * @param state
   */
  void setOrthographicCamera(bool state);

  /**
   * @brief TODO()
   *
   * @param X
   * @param Y
   */
  void setScale(qreal X, qreal Y);
};
