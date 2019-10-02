#include <osg/ArgumentParser>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

#include "ColorSet.hpp"
#include "StlFile.hpp"
#include "StlMesh.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }

  osg::ref_ptr<osg::Group> root = new osg::Group;

  boost::filesystem::path input(pathString);

  ColorSet colorSet;
  int numModels = 0;

  if (is_directory(input)) {
    boost::filesystem::directory_iterator end;
    for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
      boost::filesystem::path filePath = it->path();
      if (filePath.extension().string() == ".stl") {
        numModels++;
        stl_file::StlFile solid(filePath.string());
        std::cout << solid.name << ": " << solid.vertices.size()
                  << " vertices, " << solid.normals.size() << " normals"
                  << std::endl;
        root->addChild(new StlMesh(solid, colorSet.getColor(numModels)));
      }
    }
  } else {
    stl_file::StlFile solid(input.string());
    std::cout << solid.name << ": " << solid.vertices.size() << " vertices, "
              << solid.normals.size() << " normals" << std::endl;
    root->addChild(new StlMesh(solid, colorSet.getColor(1)));
  }
  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}