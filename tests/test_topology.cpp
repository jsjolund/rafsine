#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osg/ArgumentParser>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <glm/vec3.hpp>

#include "PartitionMesh.hpp"

int main(int argc, char **argv)
{
  osg::ArgumentParser args(&argc, argv);

  int nx = 512, ny = 512, nz = 512;
  int divisions = 0;

  int value;
  if (args.read("-nx", value))
  {
    nx = value;
  }
  if (args.read("-ny", value))
  {
    ny = value;
  }
  if (args.read("-nz", value))
  {
    nz = value;
  }
  if (args.read("-d", value))
  {
    divisions = value;
  }

  osg::ref_ptr<PartitionMesh> mesh = new PartitionMesh(nx, ny, nz, divisions);
  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(mesh);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0.5, 0.5, 0.5, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}