#pragma once

#include <osg/Node>
#include <osg/NodeCallback>
#include <osg/NodeVisitor>

#include <cuda_profiler_api.h>

#include "CFDHud.hpp"
#include "CFDScene.hpp"
#include "DomainData.hpp"
#include "PickHandler.hpp"
#include "QtOSGWidget.hpp"
#include "SimulationWorker.hpp"
#include "SliceRender.hpp"

/**
 * @brief This callback is executed when the camera is updated
 *
 */
class CameraUpdateCallback : public osg::NodeCallback {
 public:
  osg::ref_ptr<osg::Camera> m_camera;
  osg::ref_ptr<osg::PositionAttitudeTransform> m_axes;

  CameraUpdateCallback(osg::ref_ptr<osg::Camera> camera,
                       osg::ref_ptr<osg::PositionAttitudeTransform> axes)
      : m_camera(camera), m_axes(axes) {}

 protected:
  virtual void operator()(osg::Node *node, osg::NodeVisitor *nv) {
    // Set the axis arrows to correct attitude
    osg::Quat q = m_camera->getViewMatrix().getRotate();
    m_axes->setAttitude(q);
    traverse(node, nv);
  }
};

/**
 * @brief Binds the OSG 3D visualization scene with a QtOSGWidget
 *
 */
class CFDWidget : public QtOSGWidget {
 private:
  osg::ref_ptr<osg::Group> m_root;
  CFDScene *m_scene;

  /**
   * @brief This class handles movement of slices on key press
   *
   */
  class CFDKeyboardHandler : public osgGA::GUIEventHandler {
   private:
    CFDWidget *m_widget;

   public:
    int m_sliceXdir, m_sliceYdir, m_sliceZdir;

    explicit CFDKeyboardHandler(CFDWidget *widget);

    virtual bool handle(const osgGA::GUIEventAdapter &ea,
                        osgGA::GUIActionAdapter &aa, osg::Object *,
                        osg::NodeVisitor *);
    virtual bool handle(osgGA::Event *event, osg::Object *object,
                        osg::NodeVisitor *nv);
    virtual bool handle(const osgGA::GUIEventAdapter &ea,
                        osgGA::GUIActionAdapter &aa);
  };
  CFDKeyboardHandler *m_keyboardHandle;
  SimulationWorker *m_simWorker;

 public:
  /**
   * @brief Construct a new CFDWidget for QT5
   *
   * @param worker A simulation thread worker
   * @param scaleX OpenGL scaling factor for display and mouse input
   * @param scaleY
   * @param parent The parent widget
   */
  CFDWidget(SimulationWorker *worker, qreal scaleX = 1, qreal scaleY = 1,
            QWidget *parent = 0);
  /**
   * @brief Update the slice positions if they should change
   *
   */
  void updateSlicePositions();
  /**
   * @brief Get the Scene object
   *
   * @return CFDScene*
   */
  inline CFDScene *getScene() { return m_scene; }
  virtual void paintGL();
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
};
