#pragma once

#include <cuda_profiler_api.h>

#include "QtOSGWidget.hpp"
#include "SliceRender.hpp"
#include "CFDScene.hpp"
#include "DomainData.hpp"
#include "SimulationThread.hpp"
#include "PickHandler.hpp"

class CFDWidget : public QtOSGWidget
{
private:
  osg::ref_ptr<osg::Group> m_root;
  CFDScene *m_scene;
  osg::Timer m_timer;
  osg::Timer_t m_lastTime;

  class CFDKeyboardHandler : public osgGA::GUIEventHandler
  {
  private:
    CFDWidget *m_widget;

  public:
    int m_sliceXdir, m_sliceYdir, m_sliceZdir;

    CFDKeyboardHandler(CFDWidget *widget);

    virtual bool handle(const osgGA::GUIEventAdapter &ea,
                        osgGA::GUIActionAdapter &aa,
                        osg::Object *, osg::NodeVisitor *);
    virtual bool handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv);
    virtual bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa);
  };
  CFDKeyboardHandler *m_keyboardHandle;

public:
  SimulationThread *m_simThread;

  CFDWidget(SimulationThread *thread, qreal scaleX, qreal scaleY, QWidget *parent = 0);

  void updateSlicePositions();
  virtual void paintGL();
  virtual void initializeGL();

  inline CFDScene *getScene() { return m_scene; };
};
