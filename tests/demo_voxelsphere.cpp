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

#include "Vector3.hpp"

class VoxelSphere {
 private:
  const unsigned int m_n;
  std::vector<bool> m_grid;
  std::vector<vector3<int>> m_normals;

  unsigned int idx(int x, int y, int z) { return x + y * m_n + z * m_n * m_n; }

  unsigned int idxn(int x, int y, int z) {
    return idx(x + m_n / 2, y + m_n / 2, z + m_n / 2);
  }

  void fill(int x, int y, int z) { m_grid.at(idxn(x, y, z)) = true; }

  void fillSigns(int x, int y, int z) {
    fill(x, y, z);
    for (;;) {
      if ((z = -z) >= 0) {
        if ((y = -y) >= 0) {
          if ((x = -x) >= 0) { break; }
        }
      }
      fill(x, y, z);
    }
  }

  void fillAll(int x, int y, int z) {
    fillSigns(x, y, z);
    if (z > y) { fillSigns(x, z, y); }
    if (z > x && z > y) { fillSigns(z, y, x); }
  }

 protected:
  bool get(unsigned int x, unsigned int y, unsigned int z) {
    try {
      return m_grid.at(idx(x, y, z));
    } catch (const std::exception& e) { return false; }
  }
  vector3<int> getNormal(unsigned int x, unsigned int y, unsigned int z) {
    return m_normals.at(idx(x, y, z));
  }
  bool getFromOrigin(int x, int y, int z) {
    try {
      return m_grid.at(idxn(x, y, z));
    } catch (const std::exception& e) { return false; }
  }
  unsigned int getSizeX() { return m_n; }
  unsigned int getSizeY() { return m_n; }
  unsigned int getSizeZ() { return m_n; }

 public:
  explicit VoxelSphere(float R)
      : m_n(floor(R) * 2 + 1),
        m_grid(m_n * m_n * m_n),
        m_normals(m_n * m_n * m_n) {
    std::fill(m_grid.begin(), m_grid.end(), false);
    std::fill(m_normals.begin(), m_normals.end(), vector3<int>(0, 0, 0));

    const int maxR2 = floor(R * R);
    int zx = floor(R);
    for (int x = 0;; ++x) {
      while (x * x + zx * zx > maxR2 && zx >= x) --zx;
      if (zx < x) break;
      int z = zx;
      for (int y = 0;; ++y) {
        while (x * x + y * y + z * z > maxR2 && z >= x && z >= y) --z;
        if (z < x || z < y) break;
        fillAll(x, y, z);
      }
    }
    std::vector<bool> cornerGrid(m_n * m_n * m_n);
    const int r = floor(R) + 1;
    for (int x = -r; x < 0; x++)
      for (int y = -r; y < r; y++)
        for (int z = -r; z < r; z++) {
          if (getFromOrigin(x, y, z) && (getFromOrigin(x + 1, y + 1, z) ||
                                         getFromOrigin(x + 1, y - 1, z) ||
                                         getFromOrigin(x + 1, y, z + 1) ||
                                         getFromOrigin(x + 1, y, z - 1)))
            cornerGrid.at(idxn(x + 1, y, z)) = true;
        }
    for (int x = 0; x < r; x++)
      for (int y = -r; y < r; y++)
        for (int z = -r; z < r; z++) {
          if (getFromOrigin(x, y, z) && (getFromOrigin(x - 1, y + 1, z) ||
                                         getFromOrigin(x - 1, y - 1, z) ||
                                         getFromOrigin(x - 1, y, z + 1) ||
                                         getFromOrigin(x - 1, y, z - 1)))
            cornerGrid.at(idxn(x - 1, y, z)) = true;
        }
    for (int x = -r; x < r; x++)
      for (int y = -r; y < 0; y++)
        for (int z = -r; z < r; z++) {
          if (getFromOrigin(x, y, z) && (getFromOrigin(x + 1, y + 1, z) ||
                                         getFromOrigin(x - 1, y + 1, z) ||
                                         getFromOrigin(x, y + 1, z + 1) ||
                                         getFromOrigin(x, y + 1, z - 1)))
            cornerGrid.at(idxn(x, y + 1, z)) = true;
        }
    for (int x = -r; x < r; x++)
      for (int y = 0; y < r; y++)
        for (int z = -r; z < r; z++) {
          if (getFromOrigin(x, y, z) && (getFromOrigin(x + 1, y - 1, z) ||
                                         getFromOrigin(x - 1, y - 1, z) ||
                                         getFromOrigin(x, y - 1, z + 1) ||
                                         getFromOrigin(x, y - 1, z - 1)))
            cornerGrid.at(idxn(x, y - 1, z)) = true;
        }
    for (int x = -r; x < r; x++)
      for (int y = -r; y < r; y++)
        for (int z = -r; z < 0; z++) {
          if (getFromOrigin(x, y, z) && (getFromOrigin(x, y + 1, z + 1) ||
                                         getFromOrigin(x, y - 1, z + 1) ||
                                         getFromOrigin(x + 1, y, z + 1) ||
                                         getFromOrigin(x - 1, y, z + 1)))
            cornerGrid.at(idxn(x, y, z + 1)) = true;
        }
    for (int x = -r; x < r; x++)
      for (int y = -r; y < r; y++)
        for (int z = 0; z < r; z++) {
          if (getFromOrigin(x, y, z) && (getFromOrigin(x, y + 1, z - 1) ||
                                         getFromOrigin(x, y - 1, z - 1) ||
                                         getFromOrigin(x + 1, y, z - 1) ||
                                         getFromOrigin(x - 1, y, z - 1)))
            cornerGrid.at(idxn(x, y, z - 1)) = true;
        }

    // for (unsigned int x = 0; x < m_n; x++)
    //   for (unsigned int y = 0; y < m_n; y++)
    //     for (unsigned int z = 0; z < m_n; z++) {
    //       if (cornerGrid.at(idx(x, y, z))) m_grid.at(idx(x, y, z)) = true;
    //     }

    // for (int x = -r; x <= 0; x++)
    //   for (int y = -r; y < r; y++)
    //     for (int z = -r; z < r; z++) {
    //       if (getFromOrigin(x, y, z))
    //         m_normals.at(idxn(x, y, z)) =
    //             m_normals.at(idxn(x, y, z)) + vector3<int>(-1, 0, 0);
    //     }
    // for (int x = 0; x <= r; x++)
    //   for (int y = -r; y < r; y++)
    //     for (int z = -r; z < r; z++) {
    //       if (getFromOrigin(x, y, z))
    //         m_normals.at(idxn(x, y, z)) =
    //             m_normals.at(idxn(x, y, z)) + vector3<int>(1, 0, 0);
    //     }
    // for (int x = -r; x < r; x++)
    //   for (int y = -r; y < 0; y++)
    //     for (int z = -r; z < r; z++) {
    //       if (getFromOrigin(x, y, z))
    //         m_normals.at(idxn(x, y, z)) =
    //             m_normals.at(idxn(x, y, z)) + vector3<int>(0, -1, 0);
    //     }
    // for (int x = -r; x < r; x++)
    //   for (int y = 0; y <= r; y++)
    //     for (int z = -r; z < r; z++) {
    //       if (getFromOrigin(x, y, z))
    //         m_normals.at(idxn(x, y, z)) =
    //             m_normals.at(idxn(x, y, z)) + vector3<int>(0, 1, 0);
    //     }
    // for (int x = -r; x < r; x++)
    //   for (int y = -r; y < r; y++)
    //     for (int z = -r; z < 0; z++) {
    //       if (getFromOrigin(x, y, z))
    //         m_normals.at(idxn(x, y, z)) =
    //             m_normals.at(idxn(x, y, z)) + vector3<int>(0, 0, -1);
    //     }
    // for (int x = -r; x < r; x++)
    //   for (int y = -r; y < r; y++)
    //     for (int z = 0; z <= r; z++) {
    //       if (getFromOrigin(x, y, z))
    //         m_normals.at(idxn(x, y, z)) =
    //             m_normals.at(idxn(x, y, z)) + vector3<int>(0, 0, 1);
    //     }
  }
};

class OSGVoxelSphere : VoxelSphere {
 private:
  const osg::Vec4 m_red;
  const osg::Vec4 m_green;
  const osg::Vec4 m_blue;
  const osg::Vec4 m_gray;

 public:
  osg::ref_ptr<osg::Group> m_root;

  void add(osg::ref_ptr<osg::Box> box, osg::Vec4 color) {
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable();
    sd->setColor(color);
    sd->setShape(box);
    m_root->addChild(sd);
  }

  explicit OSGVoxelSphere(float R)
      : VoxelSphere(R),
        m_red(1, 0, 0, 1),
        m_green(0, 1, 0, 1),
        m_blue(0, 0, 1, 1),
        m_gray(0.5, 0.5, 0.5, 0.5),
        m_root(new osg::Group) {
    for (unsigned int x = 0; x < getSizeX(); x++)
      for (unsigned int y = 0; y < getSizeY(); y++)
        for (unsigned int z = 0; z < getSizeZ(); z++) {
          if (get(x, y, z)) {
            vector3<int> normal = getNormal(x, y, z);
            if (normal.x() > 0) {
              add(new osg::Box(osg::Vec3f(x + 0.25, y, z), 0.75, 0.1, 0.1),
                  m_red);
            }
            if (normal.y() > 0) {
              add(new osg::Box(osg::Vec3f(x, y + 0.25, z), 0.1, 0.75, 0.1),
                  m_green);
            }
            if (normal.z() > 0) {
              add(new osg::Box(osg::Vec3f(x, y, z + 0.25), 0.1, 0.1, 0.75),
                  m_blue);
            }
            if (normal.x() < 0) {
              add(new osg::Box(osg::Vec3f(x - 0.25, y, z), 0.75, 0.1, 0.1),
                  m_red);
            }
            if (normal.y() < 0) {
              add(new osg::Box(osg::Vec3f(x, y - 0.25, z), 0.1, 0.75, 0.1),
                  m_green);
            }
            if (normal.z() < 0) {
              add(new osg::Box(osg::Vec3f(x, y, z - 0.25), 0.1, 0.1, 0.75),
                  m_blue);
            }
            add(new osg::Box(osg::Vec3f(x, y, z), 1), m_gray);
          }
        }
  }
};

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  float radius = 12.7;
  // float radius = 5.7;
  float value;
  if (args.read("-r", value)) { radius = value; }

  OSGVoxelSphere sphere(radius);

  osgViewer::Viewer viewer;
  osg::StateSet* stateset = viewer.getCamera()->getOrCreateStateSet();
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  viewer.setSceneData(sphere.m_root);
  viewer.setUpViewInWindow(400, 400, 800, 600);
  viewer.addEventHandler(new osgViewer::StatsHandler);
  return viewer.run();
}
