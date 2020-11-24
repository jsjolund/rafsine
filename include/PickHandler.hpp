// #pragma once

// #include <osgViewer/Viewer>

// #include <sstream>

// #include "CFDScene.hpp"

// /**
//  * @brief Class to handle mouse click events on the 3D voxel visualization
//  *
//  */
// class PickHandler : public osgGA::GUIEventHandler {
//  private:
//   CFDScene* m_scene;

//  protected:
//   ~PickHandler() {}

//  public:
//   /**
//    * @brief Construct a new Pick Handler
//    *
//    * @param scene The CFD scene to pick objects in
//    */
//   explicit PickHandler(CFDScene* scene);
//   /**
//    * @brief Executed when user (double) clicks on the voxel model
//    *
//    * @param view The view
//    * @param ea Adapter for the click event
//    * @return true A valid target was picked
//    * @return false No valid target was picked
//    */
//   bool pick(osgViewer::View* view, const osgGA::GUIEventAdapter& ea);
//   /**
//    * @brief Filters input events and calls the picking function
//    *
//    * @param ea  Adapter for the click event
//    * @param aa Adapter containing the view
//    * @return true The event was handled
//    * @return false The event was not handled
//    */
//   bool handle(const osgGA::GUIEventAdapter& ea,
//               osgGA::GUIActionAdapter& aa,
//               osg::Object*,
//               osg::NodeVisitor*) override;
//   /**
//    * @brief Alternative input event handle
//    *
//    * @param event
//    * @param object
//    * @param nv
//    * @return true
//    * @return false
//    */
//   bool handle(osgGA::Event* event,
//               osg::Object* object,
//               osg::NodeVisitor* nv) override;
//   /**
//    * @brief Alternative input event handle
//    *
//    * @return true
//    * @return false
//    */
//   bool handle(const osgGA::GUIEventAdapter&, osgGA::GUIActionAdapter&)
//   override;
// };
