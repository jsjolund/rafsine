#include "InputEventHandler.hpp"

bool InputEventHandler::handle(osgGA::Event* event,
                               osg::Object* object,
                               osg::NodeVisitor* nv) {
  return handle(*(event->asGUIEventAdapter()),
                *(nv->asEventVisitor()->getActionAdapter()), object, nv);
}

bool InputEventHandler::handle(const osgGA::GUIEventAdapter& ea,
                               osgGA::GUIActionAdapter& aa) {
  return handle(ea, aa, NULL, NULL);
}

bool InputEventHandler::handle(const osgGA::GUIEventAdapter& ea,
                               osgGA::GUIActionAdapter&,
                               osg::Object*,
                               osg::NodeVisitor*) {
  typedef osgGA::GUIEventAdapter::KeySymbol osgKey;

  switch (ea.getEventType()) {
    case (osgGA::GUIEventAdapter::KEYDOWN): return keyDown(ea.getKey());
    case (osgGA::GUIEventAdapter::KEYUP): return keyUp(ea.getKey());
    default: return false;
  }
}
