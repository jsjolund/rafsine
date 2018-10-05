#pragma once

#include <osgViewer/Viewer>

#include <sstream>

#include "CFDScene.hpp"

// class to handle events with a pick
class PickHandler : public osgGA::GUIEventHandler {
 private:
  CFDScene *m_scene;

 public:
  explicit PickHandler(CFDScene *scene);
  ~PickHandler() {}
  bool pick(osgViewer::View *view, const osgGA::GUIEventAdapter &ea);
  bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa,
              osg::Object *, osg::NodeVisitor *) override;
  bool handle(osgGA::Event *event, osg::Object *object,
              osg::NodeVisitor *nv) override;
  bool handle(const osgGA::GUIEventAdapter &,
              osgGA::GUIActionAdapter &) override;
};
