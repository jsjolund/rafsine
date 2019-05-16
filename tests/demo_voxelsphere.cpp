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

// I don't believe running the midpoint circle algorithm on each layer will give
// the desired results once you reach the poles, as you will have gaps in the
// surface where LEDs are not lit. This may give the result you want, however,
// so that would be up to aesthetics. This post is based on using the midpoint
// circle algorithm to determine the radius of the layers through the middle two
// vertical octants, and then when drawing each of those circles also setting
// the points for the polar octants.

// I think based on @Nick Udall's comment and answer here using the circle
// algorithm to determine radius of your horizontal slice will work with a
// modification I proposed in a comment on his answer. The circle algorithm
// should be modified to take as an input an initial error, and also draw the
// additional points for the polar octants.

// Draw the standard circle algorithm points at y0 + y1 and y0 - y1: x0 +/- x,
// z0 +/- z, y0 +/- y1, x0 +/- z, z0 +/- x, y0 +/- y1, total 16 points. This
// forms the bulk of the vertical of the sphere. Additionally draw the points x0
// +/- y1, z0 +/- x, y0 +/- z and x0 +/- x, z0 +/- y1, y0 +/- z, total 16
// points, which will form the polar caps for the sphere. By passing the outer
// algorithm's error into the circle algorithm, it will allow for sub-voxel
// adjustment of each layer's circle. Without passing the error into the inner
// algorithm, the equator of the circle will be approximated to a cylinder, and
// each approximated sphere face on the x, y, and z axes will form a square.
// With the error included, each face given a large enough radius will be
// approximated as a filled circle.

// The following code is modified from Wikipedia's Midpoint circle algorithm.
// The DrawCircle algorithm has the nomenclature changed to operate in the
// xz-plane, addition of the third initial point y0, the y offset y1, and
// initial error error0. DrawSphere was modified from the same function to take
// the third initial point y0 and calls DrawCircle rather than DrawPixel

// For a sphere of radius 4 (which actually requires 9x9x9), this would run
// three iterations of the DrawCircle routine, with the first drawing a typical
// radius 4 circle (three steps), the second drawing a radius 4 circle with
// initial error of 0 (also three steps), and then the third drawing a radius 3
// circle with initial error 0 (also three steps). That ends up being nine
// calculated points, drawing 32 pixels each. That makes 32 (points per circle)
// x 3 (add or subtract operations per point) + 6 (add, subtract, shift
// operations per iteration) = 102 add, subtract, or shift operations per
// calculated point. In this example, that's 3 points for each circle = 306
// operations per layer. The radius algorithm also adds 6 operations per layer
// and iterates 3 times, so 306 + 6 * 3 = 936 basic arithmetic operations for
// the example radius of 4. The cost here is that you will repeatedly set some
// pixels without additional condition checks (i.e. x = 0, y = 0, or z = 0), so
// if your I/O is slow you may be better off adding the condition checks.
// Assuming all LEDs were cleared at the start, the example circle would set 288
// LEDs, while there are many fewer LEDs that would actually be lit due to
// repeat sets.

// It looks like this would perform better than the bruteforce method for all
// spheres that would fit in the 8x8x8 grid, but the bruteforce method would
// have consistent timing regardless of radius, while this method will slow down
// when drawing large radius spheres where only part will be displayed. As the
// display cube increases in resolution, however, this algorithm timing will
// stay consistent while bruteforce will increase.

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
