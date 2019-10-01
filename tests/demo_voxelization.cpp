
#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

#include "StlFile.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }

  boost::filesystem::path input(pathString);
  boost::filesystem::directory_iterator end;
  for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
    boost::filesystem::path filePath = it->path();
    if (filePath.extension().string() == ".stl") {
      stl_file::StlFile solid(filePath.string());
      std::cout << solid.name << ": " << solid.vertices.size() << " vertices, "
                << solid.normals.size() << " normals" << std::endl;
    }
  }

  // osg::ref_ptr<osg::Group> root = new osg::Group;
  // root->addChild(mesh);
  // osgViewer::Viewer viewer;
  // viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  // viewer.setSceneData(root);
  // viewer.setUpViewInWindow(400, 400, 800, 600);
  // viewer.addEventHandler(new osgViewer::StatsHandler);
  // return viewer.run();
  return 0;
}
