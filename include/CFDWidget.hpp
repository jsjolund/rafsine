#pragma once

#include <QMutex>

#include "CFDHud.hpp"
#include "CFDScene.hpp"
#include "DomainData.hpp"
#include "InputEventHandler.hpp"
#include "PickHandler.hpp"
#include "QtOSGWidget.hpp"
#include "SimulationWorker.hpp"
#include "SliceRender.hpp"

/**
 * @brief This callback is executed when the camera is updated, to orient the
 * axis model
 *
 */
class CameraUpdateCallback : public osg::NodeCallback {
 private:
  osg::ref_ptr<osg::Camera> m_camera;
  osg::ref_ptr<osg::PositionAttitudeTransform> m_axes;

 public:
  /**
   * @brief Construct a new Camera Update Callback object
   *
   * @param camera The camera to trigger callbacks
   * @param axes The axis model
   */
  CameraUpdateCallback(osg::ref_ptr<osg::Camera> camera,
                       osg::ref_ptr<osg::PositionAttitudeTransform> axes)
      : m_camera(camera), m_axes(axes) {}

 protected:
  /**
   * @brief Update the axis model attitude when camera is updated
   *
   * @param node
   * @param nv
   */
  virtual void operator()(osg::Node *node, osg::NodeVisitor *nv) {
    // Set the axis arrows to correct attitude
    osg::Quat q = m_camera->getViewMatrix().getRotate();
    if (m_axes) m_axes->setAttitude(q);
    traverse(node, nv);
  }
};

/**
 * @brief Wraps the OSG 3D visualization scene in a QtOSGWidget
 *
 */
class CFDWidget : public QtOSGWidget {
 private:
  /**
   * @brief This class handles movement of slices on key press
   *
   */
  class CFDKeyboardHandler : public InputEventHandler {
   private:
    CFDWidget *m_widget;

   public:
    int m_sliceXdir, m_sliceYdir, m_sliceZdir;

    explicit CFDKeyboardHandler(CFDWidget *widget);
    virtual bool keyDown(int key);
    virtual bool keyUp(int key);
  };

  osg::ref_ptr<osg::Group> m_root;
  CFDScene *m_scene;
  CFDKeyboardHandler *m_keyboardHandle;
  SimulationWorker *m_simWorker;
  double m_sliceMoveCounter;
  QMutex m_mutex;

 public:
  void adjustDisplayColors();
  /**
   * @brief Construct a new CFDWidget for QT5
   *
   * @param worker A simulation thread worker
   * @param scaleX OpenGL scaling factor for display and mouse input
   * @param scaleY
   * @param parent The parent widget
   */
  CFDWidget(qreal scaleX = 1, qreal scaleY = 1, QWidget *parent = NULL);

  void setSimulationWorker(SimulationWorker *simWorker);

  /**
   * @brief Get the Scene object
   *
   * @return CFDScene*
   */
  inline CFDScene *getScene() { return m_scene; }

  /**
   * @brief Draws the 3D graphics each frame
   *
   */
  void render(double deltaTime) override;

  /**
   * @brief Called when OpenGL is initialized
   *
   */
  virtual void initializeGL();

  /**
   * @brief Called when the OpenGL graphics window is resized
   *
   * @param width
   * @param height
   */
  virtual void resizeGL(int width, int height);
};
