
#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/model.stl" << std::endl;
    return -1;
  }

  boost::filesystem::path input(pathString);
  std::vector<boost::filesystem::path> filePaths;
  boost::filesystem::directory_iterator end;
  for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
    if (it->path().extension().string() == ".stl")
      filePaths.push_back(it->path());
  }
  for (boost::filesystem::path filePath : filePaths) {
    std::cout << filePath << std::endl;
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
