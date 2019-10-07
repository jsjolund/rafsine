#include <osg/ArgumentParser>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "Eigen/Geometry"

#include "ColorSet.hpp"
#include "StlMesh.hpp"
#include "StlModel.hpp"
#include "StlVoxelGeometry.hpp"
#include "VoxelMesh.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }

  boost::filesystem::path input(pathString);
  std::vector<stl_mesh::StlMesh*> meshes;

  if (is_directory(input)) {
    boost::filesystem::directory_iterator end;
    for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
      boost::filesystem::path filePath = it->path();
      if (filePath.extension().string() == ".stl") {
        meshes.push_back(new stl_mesh::StlMesh(filePath.string()));
      }
    }
  } else {
    meshes.push_back(new stl_mesh::StlMesh(input.string()));
  }

  Eigen::Matrix3f tra = Eigen::Matrix3f::Identity();
  tra.row(1).swap(tra.row(2));
  StlVoxelGeometry voxGeo(128, 118, 58, meshes, tra);
  Eigen::Vector3f min, max;
  voxGeo.getExtents(&min, &max);
  std::cout << "min=" << min.x() << ", " << min.y() << ", " << min.z() << ", "
            << "max=" << max.x() << ", " << max.y() << ", " << max.z()
            << std::endl;
  voxGeo.voxelize();

  osg::ref_ptr<osg::Group> root = new osg::Group;

  root->addChild(new VoxelMesh(voxGeo.getVoxelArray()));

  // ColorSet colorSet;
  // for (int i = 0; i < meshes.size(); i++) {
  //   stl_mesh::StlMesh* mesh = meshes.at(i);
  //   root->addChild(new StlModel(*mesh, colorSet.getColor(i + 1)));
  // }

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}
