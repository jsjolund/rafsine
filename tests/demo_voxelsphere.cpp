#include <osg/ArgumentParser>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

/*
  I think the easiest way to do this is something like the Midpoint Circle
Algorithm, extended to 3D.

First, lets figure out which blocks we want to fill. Assuming an origin in the
middle of block (0,0,0) and radius R:

We only want to fill boxes inside the sphere. Those are exactly the boxes
(x,y,z) such that x²+y²+z² <= R²; and We only want to fill boxes with a face
showing. If a box has a face showing, then at least one of its neighbors is not
in the sphere, so: (|x|+1)²+y²+z² > R² OR x²+(|y|+1)²+z² > R² OR x²+y²+(|z|+1)²
> R² It's the 2nd part that makes it tricky, but remember that (|a|+1)² = |a|² +
2|a| + 1. If, say, z is the largest coordinate of a box that is inside the
sphere, and if that box has a face showing, then the z face in particular will
be showing, because x²+y²+(|z|+1)² = x²+y²+z²+2|z|+1, and that will be at least
as big as the analogous values for x and y.

So, it's pretty easy to calculate the boxes that are 1) inside the sphere, 2)
have z as their largest coordinate, and 3) have the largest possible z value,
i.e., adding 1 to z results in a box outside the sphere. Additionally, 4) have
positive values for all x,y,z.

The coordinates of these boxes can then be reflected 24 different ways to
generate all the boxes on the surface of the sphere. Those are all 8
combinations of signs of the coordinates times all 3 choices for which axis has
the largest coordinate.

Here's how to generate the points with positive x,y,z and z largest:

 NOTE: If it matters to you, be careful when you calculate the reflections so
that you don't draw, for example, (0,y,z) and (-0,y,z), because that's the same
box twice. Also don't swap axes with the same value, because again that would
draw the same box twice (e.g., if you have (1,5,5), don't swap y and z and draw
again.

NOTE ALSO that R doesn't have to be an integer. It'll look a little nicer if you
add 0.5 to it.

Here's an example that takes all of the above into account (you need a browser
that supports webgl) https://jsfiddle.net/mtimmerm/ay2adpwb/
*/

osg::ref_ptr<osg::Group> root;

void fill(int x, int y, int z) {
  osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3f(x, y, z), 1);
  osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable();
  sd->setShape(box);
  root->addChild(sd);
}

void fillSigns(int x, int y, int z) {
  fill(x, y, z);
  for (;;) {
    if ((z = -z) >= 0) {
      if ((y = -y) >= 0) {
        if ((x = -x) >= 0) {
          break;
        }
      }
    }
    fill(x, y, z);
  }
}

void fillAll(int x, int y, int z) {
  fillSigns(x, y, z);
  if (z > y) {
    fillSigns(x, z, y);
  }
  if (z > x && z > y) {
    fillSigns(z, y, x);
  }
}

void drawSphere(float R) {
  int maxR2 = floor(R * R);
  int zx = floor(R);
  for (int x = 0;; ++x) {
    // max z for this x value.
    while (x * x + zx * zx > maxR2 && zx >= x) --zx;
    if (zx < x) break;  // with this x, z can't be largest
    int z = zx;
    for (int y = 0;; ++y) {
      while (x * x + y * y + z * z > maxR2 && z >= x && z >= y) --z;
      if (z < x || z < y) break;  // with this x and y, z can't be largest
      fillAll(x, y, z);           //... and up to 23 reflections of it
    }
  }
}

int main(int argc, char **argv) {
  osg::ArgumentParser args(&argc, argv);

  float radius = 10.0;
  float value;
  if (args.read("-r", value)) {
    radius = value;
  }

  root = new osg::Group;
  drawSphere(radius);

  osgViewer::Viewer viewer;
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}
