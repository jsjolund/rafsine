#pragma once

#include <osg/Node>
#include <osg/NodeCallback>
#include <osg/NodeVisitor>
#include <osgGA/EventVisitor>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

class InputEventHandler : public osgGA::GUIEventHandler {
 public:
  virtual bool handle(const osgGA::GUIEventAdapter& ea,
                      osgGA::GUIActionAdapter& aa,
                      osg::Object*,
                      osg::NodeVisitor*);
  virtual bool handle(osgGA::Event* event,
                      osg::Object* object,
                      osg::NodeVisitor* nv);
  virtual bool handle(const osgGA::GUIEventAdapter& ea,
                      osgGA::GUIActionAdapter& aa);

  virtual bool keyDown(int) { return false; }
  virtual bool keyUp(int) { return false; }
};
