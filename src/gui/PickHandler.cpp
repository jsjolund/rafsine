#include "PickHandler.hpp"

PickHandler::PickHandler(CFDScene* scene)
    : osgGA::GUIEventHandler(), m_scene(scene) {}

bool PickHandler::handle(const osgGA::GUIEventAdapter& ea,
                         osgGA::GUIActionAdapter& aa,
                         osg::Object*,
                         osg::NodeVisitor*) {
  if (ea.getButton() != osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON) return false;

  switch (ea.getEventType()) {
    case (osgGA::GUIEventAdapter::DOUBLECLICK): {
      osgViewer::View* view = dynamic_cast<osgViewer::View*>(&aa);
      if (view) return pick(view, ea);
      return false;
    }
    default: return false;
  }
}

bool PickHandler::handle(osgGA::Event* event,
                         osg::Object* object,
                         osg::NodeVisitor* nv) {
  return handle(*(event->asGUIEventAdapter()),
                *(nv->asEventVisitor()->getActionAdapter()), object, nv);
}

bool PickHandler::handle(const osgGA::GUIEventAdapter& ea,
                         osgGA::GUIActionAdapter& aa) {
  return handle(ea, aa, NULL, NULL);
}

bool PickHandler::pick(osgViewer::View* view,
                       const osgGA::GUIEventAdapter& ea) {
  osgUtil::LineSegmentIntersector::Intersections intersections;
  // Hide axis model to prevent clicking on it
  m_scene->setAxesVisible(false);
  // Find intersections from mouse click
  if (view->computeIntersections(ea, intersections)) {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr =
             intersections.begin();
         hitr != intersections.end(); ++hitr) {
      if (hitr->drawable.valid()) {
        if (m_scene->selectVoxel(hitr->getWorldIntersectPoint())) {
          m_scene->setAxesVisible(true);
          return true;
        }
      }
    }
  }
  // Show the axis model again
  m_scene->setAxesVisible(true);
  return false;
}
