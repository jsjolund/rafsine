#pragma once

#include "QtOSGWidget.hpp"
#include "SliceRender.hpp"
#include "CFDScene.hpp"
#include "../sim/KernelData.hpp"

class CFDKeyboardHandler;

class CFDWidget : public QtOSGWidget
{
private:
  osg::ref_ptr<osg::Group> m_root;
  CFDScene *m_scene;
  KernelData *m_kernelData;

public:
  CFDWidget(qreal scaleX, qreal scaleY, QWidget *parent = 0);

  virtual void paintGL();
  virtual void initializeGL();

  inline CFDScene *getScene() { return m_scene; };
};

class CFDKeyboardHandler : public osgGA::GUIEventHandler
{
protected:
  CFDWidget *m_widget;

public:
  CFDKeyboardHandler(CFDWidget *widget);

  virtual bool handle(const osgGA::GUIEventAdapter &ea,
                      osgGA::GUIActionAdapter &aa,
                      osg::Object *, osg::NodeVisitor *);
  virtual bool handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv);
  virtual bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa);
};