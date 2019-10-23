#include <osg/ArgumentParser>
#include <osg/Geode>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/timer.hpp>

#include "Eigen/Geometry"

#include "ColorSet.hpp"
#include "InputEventHandler.hpp"
#include "StlMesh.hpp"
#include "StlModel.hpp"
#include "StlVoxelGeometry.hpp"
#include "VoxelMesh.hpp"

namespace Axis3D {
enum Enum { Xaxis, Yaxis, Zaxis };
}

class MyKeyboardHandler : public InputEventHandler {
 private:
  osg::ref_ptr<osg::Group> m_root;
  osg::ref_ptr<VoxelMesh> m_voxMesh;
  osg::ref_ptr<osg::Geode> m_stlMesh;
  osg::Vec3i m_voxMin, m_voxMax, m_voxSize;

 public:
  explicit MyKeyboardHandler(osg::ref_ptr<osg::Group> root,
                             osg::ref_ptr<osg::Geode> stlMesh,
                             osg::ref_ptr<VoxelMesh> voxMesh,
                             osg::Vec3i min,
                             osg::Vec3i max,
                             osg::Vec3i size)
      : m_root(root),
        m_stlMesh(stlMesh),
        m_voxMesh(voxMesh),
        m_voxMin(min),
        m_voxSize(size),
        m_voxMax(max) {}

  void slice(Axis3D::Enum axis, int inc) {
    if (inc == 0) return;
    int pos;
    switch (axis) {
      case Axis3D::Xaxis:
        pos = m_voxMin.x();
        m_voxMin.x() =
            (pos + inc < m_voxSize.x() && pos + inc >= 0) ? pos + inc : pos;
        break;
      case Axis3D::Yaxis:
        pos = m_voxMin.y();
        m_voxMin.y() =
            (pos + inc < m_voxSize.y() && pos + inc >= 0) ? pos + inc : pos;
        break;
      case Axis3D::Zaxis:
        pos = m_voxMax.z();
        m_voxMax.z() =
            (pos + inc < m_voxSize.z() && pos + inc >= 0) ? pos + inc : pos;
        break;
    }
    m_voxMesh->crop(m_voxMin, m_voxMax);
  }

  virtual bool keyDown(int key) {
    typedef osgGA::GUIEventAdapter::KeySymbol osgKey;
    boost::timer t;
    switch (key) {
      case osgKey::KEY_1:
        m_voxMesh->setNodeMask(~0);
        m_stlMesh->setNodeMask(0);
        return true;
      case osgKey::KEY_2:
        m_voxMesh->setNodeMask(0);
        m_stlMesh->setNodeMask(~0);
        return true;
      case osgKey::KEY_F1:
        m_voxMesh->setPolygonMode(osg::PolygonMode::FILL);
        return true;
      case osgKey::KEY_F2:
        m_voxMesh->setPolygonMode(osg::PolygonMode::LINE);
        return true;
      case osgKey::KEY_F3:
        m_voxMesh->setPolygonMode(osg::PolygonMode::POINT);
        return true;
      case osgKey::KEY_Page_Down:
        slice(Axis3D::Zaxis, -1);
        return true;
      case osgKey::KEY_Page_Up:
        slice(Axis3D::Zaxis, 1);
        return true;
      case osgKey::KEY_End:
        slice(Axis3D::Yaxis, -1);
        return true;
      case osgKey::KEY_Home:
        slice(Axis3D::Yaxis, 1);
        return true;
      case osgKey::KEY_Delete:
        slice(Axis3D::Xaxis, -1);
        return true;
      case osgKey::KEY_Insert:
        slice(Axis3D::Xaxis, 1);
        return true;
      default:
        return false;
    }
  }
};

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }
  // Read the input file(s)
  boost::filesystem::path input(pathString);
  std::vector<std::string> stlFilePaths;
  if (is_directory(input)) {
    boost::filesystem::directory_iterator end;
    for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
      boost::filesystem::path filePath = it->path();
      if (filePath.extension().string() == ".stl")
        stlFilePaths.push_back(filePath.string());
    }
  } else {
    stlFilePaths.push_back(input.string());
  }
  // Sort input files by name
  std::sort(stlFilePaths.begin(), stlFilePaths.end());
  // Read the geometry
  std::vector<stl_mesh::StlMesh*> meshes;
  for (std::string stlFilePath : stlFilePaths) {
    meshes.push_back(new stl_mesh::StlMesh(stlFilePath));
  }

  osg::ref_ptr<osg::Group> root = new osg::Group;

  // Build voxel mesh from STL files
  std::cout << "Voxelizing..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  Eigen::Matrix3f tra = Eigen::Matrix3f::Identity();
  tra.row(1).swap(tra.row(2));
  tra.row(0).swap(tra.row(1));
  StlVoxelGeometry voxGeo(64, 59, 29, meshes, tra);
  Eigen::Vector3f min, max;
  voxGeo.getExtents(&min, &max);
  std::cout << "min=" << min.x() << ", " << min.y() << ", " << min.z() << ", "
            << "max=" << max.x() << ", " << max.y() << ", " << max.z()
            << std::endl;
  voxGeo.voxelize();

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Voxelization time: " << elapsed.count() << " s" << std::endl;

  std::cout << "Building voxel voxMesh..." << std::endl;
  start = std::chrono::high_resolution_clock::now();

  osg::ref_ptr<VoxelMesh> voxMesh = new VoxelMesh(voxGeo.getVoxelArray());

  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "Voxel voxMesh build time: " << elapsed.count() << " s"
            << std::endl;

  root->addChild(voxMesh);

  osg::Vec3i voxSize(voxMesh->getSizeX(), voxMesh->getSizeY(),
                     voxMesh->getSizeZ());
  osg::Vec3i voxMin(-1, -1, -1);
  osg::Vec3i voxMax(voxSize - osg::Vec3i(1, 1, 1));

  // Build OSG meshes from STL files
  osg::ref_ptr<osg::Geode> stlMesh = new osg::Geode;
  ColorSet colorSet;
  for (int i = 0; i < meshes.size(); i++) {
    stl_mesh::StlMesh* m = meshes.at(i);
    stlMesh->addChild(new StlModel(*m, colorSet.getColor(i + 1)));
  }
  stlMesh->setNodeMask(0);
  root->addChild(stlMesh);

  // Start viewer
  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  viewer.addEventHandler(
      new MyKeyboardHandler(root, stlMesh, voxMesh, voxMin, voxMax, voxSize));
  return viewer.run();
}
