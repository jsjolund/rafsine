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
#include "StlVoxelizer.hpp"

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/stl_directory" << std::endl;
    return -1;
  }

  osg::ref_ptr<osg::Group> root = new osg::Group;

  boost::filesystem::path input(pathString);
  std::vector<stl_mesh::StlMesh> meshes;

  if (is_directory(input)) {
    boost::filesystem::directory_iterator end;
    for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
      boost::filesystem::path filePath = it->path();
      if (filePath.extension().string() == ".stl") {
        stl_mesh::StlMesh mesh(filePath.string());
        meshes.push_back(mesh);
      }
    }
  } else {
    stl_mesh::StlMesh mesh(input.string());
    meshes.push_back(mesh);
  }

  ColorSet colorSet;
  int numModels = 0;

  for (stl_mesh::StlMesh mesh : meshes) {
    root->addChild(new StlModel(mesh, colorSet.getColor(numModels++)));
    std::cout << mesh.name << ": " << mesh.vertices.size() << " vertices, "
              << mesh.normals.size() << " normals" << std::endl;
  }

  glm::vec3 min, max;
  getExtents(meshes, &min, &max);
  std::cout << "min=" << min.x << ", " << min.y << ", " << min.z
            << ", max=" << max.x << ", " << max.y << ", " << max.z << std::endl;

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}
