#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include <osg/Geode>
#include <osg/Vec3>
#include <osg/Geometry>
#include <osg/Material>
#include <osgViewer/Viewer>
#include <osg/Math>
#include <osg/ShapeDrawable>
#include <osg/ArgumentParser>

#include <cuda.h>
#include <cuda_profiler_api.h>

#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <glm/vec3.hpp>

#include "PartitionTopology.hpp"

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

  Topology topology(nx, ny, nz, divisions);
  topology.buildMesh();

  osg::Group *root = new osg::Group;
  root->addChild(topology.m_root);

  osgViewer::Viewer viewer;
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}