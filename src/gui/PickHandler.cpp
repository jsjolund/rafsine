#include "PickHandler.hpp"

bool PickHandler::handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa, osg::Object *, osg::NodeVisitor *)
{
  if (ea.getButton() != osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)
    return false;

  switch (ea.getEventType())
  {
  case (osgGA::GUIEventAdapter::DRAG):
  {
    m_isDragging = true;
    return false;
  }
  case (osgGA::GUIEventAdapter::PUSH):
  {
    m_isDragging = false;
    return false;
  }
  case (osgGA::GUIEventAdapter::RELEASE):
  {
    if (m_isDragging)
      return false;
    osgViewer::View *view = dynamic_cast<osgViewer::View *>(&aa);
    if (view)
      pick(view, ea);
    return false;
  }
  // case (osgGA::GUIEventAdapter::KEYDOWN):
  // {
  //   if (ea.getKey() == 'c')
  //   {
  //     osgViewer::View *view = dynamic_cast<osgViewer::View *>(&aa);
  //     osg::ref_ptr<osgGA::GUIEventAdapter> event = new osgGA::GUIEventAdapter(ea);
  //     event->setX((ea.getXmin() + ea.getXmax()) * 0.5);
  //     event->setY((ea.getYmin() + ea.getYmax()) * 0.5);
  //     if (view)
  //       pick(view, *event);
  //   }
  //   return false;
  // }
  default:
    return false;
  }
}

bool PickHandler::handle(osgGA::Event *event, osg::Object *object, osg::NodeVisitor *nv)
{
  return handle(*(event->asGUIEventAdapter()), *(nv->asEventVisitor()->getActionAdapter()), object, nv);
}

bool PickHandler::handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &aa)
{
  return handle(ea, aa, NULL, NULL);
}

void PickHandler::pick(osgViewer::View *view, const osgGA::GUIEventAdapter &ea)
{
  osgUtil::LineSegmentIntersector::Intersections intersections;

  std::string gdlist = "";

  if (view->computeIntersections(ea, intersections))
  {
    for (osgUtil::LineSegmentIntersector::Intersections::iterator hitr = intersections.begin();
         hitr != intersections.end();
         ++hitr)
    {
      std::ostringstream os;
      if (!hitr->nodePath.empty() && !(hitr->nodePath.back()->getName().empty()))
      {
        // the geodes are identified by name.
        os << "Object \"" << hitr->nodePath.back()->getName() << "\"" << std::endl;
      }
      else if (hitr->drawable.valid())
      {
        os << "Object \"" << hitr->drawable->className() << "\"" << std::endl;
      }

      os << "        local coords vertex(" << hitr->getLocalIntersectPoint() << ")"
         << "  normal(" << hitr->getLocalIntersectNormal() << ")" << std::endl;
      os << "        world coords vertex(" << hitr->getWorldIntersectPoint() << ")"
         << "  normal(" << hitr->getWorldIntersectNormal() << ")" << std::endl;
      const osgUtil::LineSegmentIntersector::Intersection::IndexList &vil = hitr->indexList;
      for (unsigned int i = 0; i < vil.size(); ++i)
      {
        os << "        vertex indices [" << i << "] = " << vil[i] << std::endl;
      }
      os << "        ratio = " << hitr->ratio << std::endl;

      gdlist += os.str();
    }
  }
  std::cout << gdlist << std::endl;
}