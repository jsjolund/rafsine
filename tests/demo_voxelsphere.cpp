#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

#include <osg/ArgumentParser>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>


#include "DdQq.hpp"
#include "InputEventHandler.hpp"
#include "Vector3.hpp"
#include "VoxelObject.hpp"

class OSGVoxelSphere : VoxelSphere {
 private:
  const osg::Vec4 m_red;
  const osg::Vec4 m_green;
  const osg::Vec4 m_blue;
  const osg::Vec4 m_gray;
  const osg::Vec4 m_yellow;

 public:
  std::vector<osg::ref_ptr<osg::Node>>* m_arrows;
  std::vector<osg::ref_ptr<osg::Node>>* m_boxes;
  osg::ref_ptr<osg::Group> m_root;

  osg::ref_ptr<osg::Node> add(osg::Vec4 color,
                              float x,
                              float y,
                              float z,
                              float nx = 1,
                              float ny = 1,
                              float nz = 1) {
    osg::ref_ptr<osg::Box> box =
        new osg::Box(osg::Vec3(x, y, z) + osg::Vec3(10, 10, 10), nx, ny, nz);
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable();
    sd->setColor(color);
    sd->setShape(box);
    m_root->addChild(sd);
    return sd;
  }

  void addArrow(float x, float y, float z, D3Q7::Enum dir) {
    osg::ref_ptr<osg::Node> n;
    float o = 0.5;
    float l = 0.75;
    float w = 0.05;
    switch (dir) {
      case D3Q7::Enum::X_AXIS_POS:
        n = add(m_red, x + o, y, z, l, w, w);
        break;
      case D3Q7::Enum::X_AXIS_NEG:
        n = add(m_red, x - o, y, z, l, w, w);
        break;
      case D3Q7::Enum::Y_AXIS_POS:
        n = add(m_green, x, y + o, z, w, l, w);
        break;
      case D3Q7::Enum::Y_AXIS_NEG:
        n = add(m_green, x, y - o, z, w, l, w);
        break;
      case D3Q7::Enum::Z_AXIS_POS:
        n = add(m_blue, x, y, z + o, w, w, l);
        break;
      case D3Q7::Enum::Z_AXIS_NEG:
        n = add(m_blue, x, y, z - o, w, w, l);
        break;
      default:
        break;
    }
    m_arrows->push_back(n);
  }

  void addBox(float x, float y, float z) {
    m_boxes->push_back(add(m_gray, x, y, z, 1, 1, 1));
  }

  void addCorner(float x, float y, float z) {
    m_boxes->push_back(add(m_yellow, x, y, z, 1, 1, 1));
  }

  void addInside(float x, float y, float z) {
    m_boxes->push_back(add(m_yellow, x, y, z, 1, 1, 1));
  }

  void addOutside(float x, float y, float z) {
    m_boxes->push_back(add(m_gray, x, y, z, 0.2, 0.2, 0.2));
  }

  explicit OSGVoxelSphere(float R)
      : VoxelSphere("sphere",
                    Vector3<int>(0, 0, 0),
                    Vector3<real_t>(0, 0, 0),
                    R,
                    NaN),
        m_red(1, 0, 0, 1),
        m_green(0, 1, 0, 1),
        m_blue(0, 0, 1, 1),
        m_gray(0.5, 0.5, 0.5, 1),
        m_yellow(1, 1, 0, 1),
        m_arrows(new std::vector<osg::ref_ptr<osg::Node>>()),
        m_boxes(new std::vector<osg::ref_ptr<osg::Node>>()),
        m_root(new osg::Group) {
    for (unsigned int x = 0; x < getSizeX(); x++)
      for (unsigned int y = 0; y < getSizeY(); y++)
        for (unsigned int z = 0; z < getSizeZ(); z++) {
          SphereVoxel::Enum vox = getVoxel(x, y, z);
          if (vox == SphereVoxel::Enum::SURFACE ||
              vox == SphereVoxel::Enum::CORNER) {
            Vector3<int> normal = getNormal(x, y, z);
            if (normal.x() > 0) { addArrow(x, y, z, D3Q7::Enum::X_AXIS_POS); }
            if (normal.x() < 0) { addArrow(x, y, z, D3Q7::Enum::X_AXIS_NEG); }
            if (normal.y() > 0) { addArrow(x, y, z, D3Q7::Enum::Y_AXIS_POS); }
            if (normal.y() < 0) { addArrow(x, y, z, D3Q7::Enum::Y_AXIS_NEG); }
            if (normal.z() > 0) { addArrow(x, y, z, D3Q7::Enum::Z_AXIS_POS); }
            if (normal.z() < 0) { addArrow(x, y, z, D3Q7::Enum::Z_AXIS_NEG); }
          }
          if (vox == SphereVoxel::Enum::SURFACE) addBox(x, y, z);
          if (vox == SphereVoxel::Enum::CORNER) addBox(x, y, z);
          // if (vox == SphereVoxel::Enum::INSIDE) addInside(x, y, z);
          // if (vox == SphereVoxel::Enum::OUTSIDE) addOutside(x, y, z);
        }
  }
};

class MyKeyboardHandler : public InputEventHandler {
 private:
  std::vector<osg::ref_ptr<osg::Node>>* m_arrows;
  std::vector<osg::ref_ptr<osg::Node>>* m_boxes;

 public:
  explicit MyKeyboardHandler(std::vector<osg::ref_ptr<osg::Node>>* arrows,
                             std::vector<osg::ref_ptr<osg::Node>>* boxes)
      : m_arrows(arrows), m_boxes(boxes) {}

  virtual bool keyDown(int key) {
    typedef osgGA::GUIEventAdapter::KeySymbol osgKey;
    switch (key) {
      case osgKey::KEY_F1:
        for (size_t i = 0; i < m_arrows->size(); i++)
          m_arrows->at(i)->setNodeMask(~m_arrows->at(i)->getNodeMask());
        return true;
      case osgKey::KEY_F2:
        for (size_t i = 0; i < m_boxes->size(); i++)
          m_boxes->at(i)->setNodeMask(~m_boxes->at(i)->getNodeMask());
        return true;
      default:
        return false;
    }
  }
};

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  float radius = 12.7;
  float value;
  if (args.read("-r", value)) { radius = value; }

  auto start = std::chrono::high_resolution_clock::now();
  OSGVoxelSphere sphere(radius);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Sphere mesh build time: " << elapsed.count() << " s"
            << std::endl;

  osgViewer::Viewer viewer;
  osg::StateSet* stateset = viewer.getCamera()->getOrCreateStateSet();
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(sphere.m_root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  viewer.addEventHandler(
      new MyKeyboardHandler(sphere.m_arrows, sphere.m_boxes));
  return viewer.run();
}
