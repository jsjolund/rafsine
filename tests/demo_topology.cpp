#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <glm/vec3.hpp>

#include "PartitionMesh.hpp"

int main(int argc, char **argv) {
  osg::ArgumentParser args(&argc, argv);

  int nq = 19, nx = 64, ny = 64, nz = 64;
  int divisions = 4;

  int value;
  if (args.read("-nx", value)) {
    nx = value;
  }
  if (args.read("-ny", value)) {
    ny = value;
  }
  if (args.read("-nz", value)) {
    nz = value;
  }
  if (args.read("-d", value)) {
    divisions = value;
  }

  osg::ref_ptr<PartitionMesh> mesh =
      new PartitionMesh(nq, nx, ny, nz, divisions);

  for (Partition p : mesh->getPartitions()) std::cout << p << std::endl;

  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(mesh);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}
