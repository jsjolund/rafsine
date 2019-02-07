#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>

#include "SubLatticeMesh.hpp"

int main(int argc, char **argv) {
  osg::ArgumentParser args(&argc, argv);

  int nx = 64, ny = 64, nz = 64;
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

  osg::ref_ptr<SubLatticeMesh> mesh =
      new SubLatticeMesh(nx, ny, nz, divisions, 1.0);

  std::cout << "size=(" << nx << ", " << ny << ", " << nz
            << "), divisions=" << divisions << std::endl;
  for (SubLattice p : mesh->getSubLattices()) std::cout << p << std::endl;

  osg::ref_ptr<osg::Group> root = new osg::Group;
  root->addChild(mesh);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  return viewer.run();
}
