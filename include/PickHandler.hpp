#include <osgUtil/Optimizer>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>

#include <osgGA/TerrainManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/AnimationPathManipulator>
#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/AnimationPathManipulator>
#include <osgGA/TerrainManipulator>

#include <osg/Geode>
#include <osg/Depth>
#include <osg/Projection>
#include <osg/MatrixTransform>
#include <osg/Camera>
#include <osg/io_utils>

#include <sstream>

#include "CFDScene.hpp"

// class to handle events with a pick
class PickHandler : public osgGA::GUIEventHandler
{
private:
  CFDScene *m_scene;

public:
  PickHandler(CFDScene *scene);
  ~PickHandler() {}
  virtual void pick(osgViewer::View *view, const osgGA::GUIEventAdapter &ea);
  bool handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa, osg::Object *, osg::NodeVisitor *) override;
  bool handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv) override;
  bool handle(const osgGA::GUIEventAdapter &, osgGA::GUIActionAdapter &) override;
};
