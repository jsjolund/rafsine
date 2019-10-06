#include <osg/ArgumentParser>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <iostream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <glm/vec3.hpp>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

#include "ColorSet.hpp"
#include "StlMesh.hpp"
#include "StlModel.hpp"
#include "StlVoxelMesh.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }

  boost::filesystem::path input(pathString);
  std::vector<stl_mesh::StlMesh> meshes;

  if (is_directory(input)) {
    boost::filesystem::directory_iterator end;
    for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
      boost::filesystem::path filePath = it->path();
      if (filePath.extension().string() == ".stl") {
        meshes.push_back(stl_mesh::StlMesh(filePath.string()));
      }
    }
  } else {
    meshes.push_back(stl_mesh::StlMesh(input.string()));
  }

  StlVoxelMesh voxMesh(meshes, 256, 236, 115);
  glm::vec3 min, max;
  voxMesh.getExtents(&min, &max);
  std::cout << "min=" << min.x << ", " << min.y << ", " << min.z << ", "
            << "max=" << max.x << ", " << max.y << ", " << max.z << std::endl;

  osg::ref_ptr<osg::Group> root = new osg::Group;
  ColorSet colorSet;
  for (int i = 0; i < meshes.size(); i++) {
    stl_mesh::StlMesh mesh = meshes.at(i);
    root->addChild(new StlModel(mesh, colorSet.getColor(i)));
    std::cout << "Loaded " mesh.name << " with " << mesh.vertices.size() / 6
              << " triangles" << std::endl;
  }

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}
