#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <osg/ArgumentParser>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <vector>

#include "DdQq.hpp"
#include "InputEventHandler.hpp"
#include "Vector3.hpp"

namespace SphereVoxel {
enum Enum { INSIDE, SURFACE, CORNER, OUTSIDE };
}

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

class VoxelSphere {
 private:
  const unsigned int m_n;
  std::vector<SphereVoxel::Enum> m_grid;
  std::vector<vector3<int>> m_normals;

  unsigned int idx(int x, int y, int z) { return x + y * m_n + z * m_n * m_n; }

  unsigned int idxn(int x, int y, int z) {
    return idx(x + m_n / 2, y + m_n / 2, z + m_n / 2);
  }

  void fill(const int x, const int y, const int z) {
    m_grid.at(idxn(x, y, z)) = SphereVoxel::Enum::SURFACE;
  }

  void fillInside(const int x, const int y, const int z) {
    int ax = abs(x);
    int ay = abs(y);
    int az = abs(z);
    int bx = -ax;
    int by = -ay;
    int bz = -az;
    for (int ix = ax; ix >= bx; ix--) {
      for (int iy = ay; iy >= by; iy--) {
        for (int iz = az; iz >= bz; iz--) {
          if (m_grid.at(idxn(ix, iy, iz)) != SphereVoxel::Enum::SURFACE)
            m_grid.at(idxn(ix, iy, iz)) = SphereVoxel::Enum::INSIDE;
        }
      }
    }
  }

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
    fillInside(x, y, z);
  }

  void fillAll(int x, int y, int z) {
    fillSigns(x, y, z);
    if (z > y) { fillSigns(x, z, y); }
    if (z > x && z > y) { fillSigns(z, y, x); }
  }

 protected:
  SphereVoxel::Enum get(unsigned int x, unsigned int y, unsigned int z) {
    return m_grid.at(idx(x, y, z));
  }

  vector3<int> getNormal(unsigned int x, unsigned int y, unsigned int z) {
    return m_normals.at(idx(x, y, z));
  }

  unsigned int getSizeX() { return m_n; }
  unsigned int getSizeY() { return m_n; }
  unsigned int getSizeZ() { return m_n; }

 public:
  explicit VoxelSphere(float R)
      : m_n(floor(R) * 2 + 2),
        m_grid(m_n * m_n * m_n),
        m_normals(m_n * m_n * m_n) {
    std::fill(m_grid.begin(), m_grid.end(), SphereVoxel::Enum::OUTSIDE);
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
    std::vector<SphereVoxel::Enum> cornerGrid(m_n * m_n * m_n);
    std::fill(cornerGrid.begin(), cornerGrid.end(), SphereVoxel::Enum::OUTSIDE);
    for (unsigned int x = 0; x < m_n; x++)
      for (unsigned int y = 0; y < m_n; y++)
        for (unsigned int z = 0; z < m_n; z++) {
          if (get(x, y, z) == SphereVoxel::Enum::INSIDE) {
            int adjacent = 0;
            if (get(x + 1, y, z) == SphereVoxel::Enum::SURFACE) adjacent++;
            if (get(x - 1, y, z) == SphereVoxel::Enum::SURFACE) adjacent++;
            if (get(x, y + 1, z) == SphereVoxel::Enum::SURFACE) adjacent++;
            if (get(x, y - 1, z) == SphereVoxel::Enum::SURFACE) adjacent++;
            if (get(x, y, z + 1) == SphereVoxel::Enum::SURFACE) adjacent++;
            if (get(x, y, z - 1) == SphereVoxel::Enum::SURFACE) adjacent++;
            if (adjacent > 1)
              cornerGrid.at(idx(x, y, z)) = SphereVoxel::Enum::CORNER;
          }
        }

    for (unsigned int x = 0; x < m_n; x++)
      for (unsigned int y = 0; y < m_n; y++)
        for (unsigned int z = 0; z < m_n; z++) {
          if (cornerGrid.at(idx(x, y, z)) == SphereVoxel::Enum::CORNER)
            m_grid.at(idx(x, y, z)) = SphereVoxel::Enum::CORNER;
        }

    for (unsigned int x = 0; x < m_n; x++)
      for (unsigned int y = 0; y < m_n; y++)
        for (unsigned int z = 0; z < m_n; z++)
          if (get(x, y, z) == SphereVoxel::Enum::SURFACE) {
            try {
              if (get(x + 1, y, z) == SphereVoxel::Enum::OUTSIDE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(1, 0, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(1, 0, 0);
            }
            try {
              if (get(x - 1, y, z) == SphereVoxel::Enum::OUTSIDE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(-1, 0, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(-1, 0, 0);
            }
            try {
              if (get(x, y + 1, z) == SphereVoxel::Enum::OUTSIDE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, 1, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, 1, 0);
            }
            try {
              if (get(x, y - 1, z) == SphereVoxel::Enum::OUTSIDE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, -1, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, -1, 0);
            }
            try {
              if (get(x, y, z + 1) == SphereVoxel::Enum::OUTSIDE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, 1);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, 1);
            }
            try {
              if (get(x, y, z - 1) == SphereVoxel::Enum::OUTSIDE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, -1);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, -1);
            }
          } else if (get(x, y, z) == SphereVoxel::Enum::CORNER) {
            try {
              if (get(x + 1, y, z) == SphereVoxel::Enum::SURFACE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(1, 0, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(1, 0, 0);
            }
            try {
              if (get(x - 1, y, z) == SphereVoxel::Enum::SURFACE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(-1, 0, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(-1, 0, 0);
            }
            try {
              if (get(x, y + 1, z) == SphereVoxel::Enum::SURFACE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, 1, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, 1, 0);
            }
            try {
              if (get(x, y - 1, z) == SphereVoxel::Enum::SURFACE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, -1, 0);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, -1, 0);
            }
            try {
              if (get(x, y, z + 1) == SphereVoxel::Enum::SURFACE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, 1);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, 1);
            }
            try {
              if (get(x, y, z - 1) == SphereVoxel::Enum::SURFACE) {
                m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, -1);
              }
            } catch (const std::exception e) {
              m_normals.at(idx(x, y, z)) += vector3<int>(0, 0, -1);
            }
          }
  }
};

class OSGVoxelSphere : VoxelSphere {
 private:
  const osg::Vec4 m_red;
  const osg::Vec4 m_green;
  const osg::Vec4 m_blue;
  const osg::Vec4 m_gray;
  const osg::Vec4 m_yellow;

 public:
  std::vector<osg::ref_ptr<osg::Node>>* m_arrows;
  std::vector<osg::ref_ptr<osg::Node>>* m_boxes;
  osg::ref_ptr<osg::Group> m_root;

  osg::ref_ptr<osg::Node> add(osg::Vec4 color,
                              float x,
                              float y,
                              float z,
                              float nx = 1,
                              float ny = 1,
                              float nz = 1) {
    osg::ref_ptr<osg::Box> box =
        new osg::Box(osg::Vec3(x, y, z) + osg::Vec3(10, 10, 10), nx, ny, nz);
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable();
    sd->setColor(color);
    sd->setShape(box);
    m_root->addChild(sd);
    return sd;
  }

  void addArrow(float x, float y, float z, D3Q7::Enum dir) {
    osg::ref_ptr<osg::Node> n;
    float o = 0.5;
    float l = 0.75;
    float w = 0.05;
    switch (dir) {
      case D3Q7::Enum::X_AXIS_POS:
        n = add(m_red, x + o, y, z, l, w, w);
        break;
      case D3Q7::Enum::X_AXIS_NEG:
        n = add(m_red, x - o, y, z, l, w, w);
        break;
      case D3Q7::Enum::Y_AXIS_POS:
        n = add(m_green, x, y + o, z, w, l, w);
        break;
      case D3Q7::Enum::Y_AXIS_NEG:
        n = add(m_green, x, y - o, z, w, l, w);
        break;
      case D3Q7::Enum::Z_AXIS_POS:
        n = add(m_blue, x, y, z + o, w, w, l);
        break;
      case D3Q7::Enum::Z_AXIS_NEG:
        n = add(m_blue, x, y, z - o, w, w, l);
        break;
      default:
        break;
    }
    m_arrows->push_back(n);
  }

  void addBox(float x, float y, float z) {
    m_boxes->push_back(add(m_gray, x, y, z, 1, 1, 1));
  }

  void addCorner(float x, float y, float z) {
    m_boxes->push_back(add(m_yellow, x, y, z, 1, 1, 1));
  }

  void addInside(float x, float y, float z) {
    m_boxes->push_back(add(m_yellow, x, y, z, 1, 1, 1));
  }

  void addOutside(float x, float y, float z) {
    m_boxes->push_back(add(m_gray, x, y, z, 0.2, 0.2, 0.2));
  }

  explicit OSGVoxelSphere(float R)
      : VoxelSphere(R),
        m_red(1, 0, 0, 1),
        m_green(0, 1, 0, 1),
        m_blue(0, 0, 1, 1),
        m_gray(0.5, 0.5, 0.5, 1),
        m_yellow(1, 1, 0, 1),
        m_arrows(new std::vector<osg::ref_ptr<osg::Node>>()),
        m_boxes(new std::vector<osg::ref_ptr<osg::Node>>()),
        m_root(new osg::Group) {
    for (unsigned int x = 0; x < getSizeX(); x++)
      for (unsigned int y = 0; y < getSizeY(); y++)
        for (unsigned int z = 0; z < getSizeZ(); z++) {
          SphereVoxel::Enum vox = get(x, y, z);
          if (vox == SphereVoxel::Enum::SURFACE ||
              vox == SphereVoxel::Enum::CORNER) {
            vector3<int> normal = getNormal(x, y, z);
            if (normal.x() > 0) { addArrow(x, y, z, D3Q7::Enum::X_AXIS_POS); }
            if (normal.x() < 0) { addArrow(x, y, z, D3Q7::Enum::X_AXIS_NEG); }
            if (normal.y() > 0) { addArrow(x, y, z, D3Q7::Enum::Y_AXIS_POS); }
            if (normal.y() < 0) { addArrow(x, y, z, D3Q7::Enum::Y_AXIS_NEG); }
            if (normal.z() > 0) { addArrow(x, y, z, D3Q7::Enum::Z_AXIS_POS); }
            if (normal.z() < 0) { addArrow(x, y, z, D3Q7::Enum::Z_AXIS_NEG); }
          }
          if (vox == SphereVoxel::Enum::SURFACE) addBox(x, y, z);
          if (vox == SphereVoxel::Enum::CORNER) addBox(x, y, z);
          // if (vox == SphereVoxel::Enum::INSIDE) addInside(x, y, z);
          if (vox == SphereVoxel::Enum::OUTSIDE) addOutside(x, y, z);
        }
  }
};

class MyKeyboardHandler : public InputEventHandler {
 private:
  std::vector<osg::ref_ptr<osg::Node>>* m_arrows;
  std::vector<osg::ref_ptr<osg::Node>>* m_boxes;

 public:
  explicit MyKeyboardHandler(std::vector<osg::ref_ptr<osg::Node>>* arrows,
                             std::vector<osg::ref_ptr<osg::Node>>* boxes)
      : m_arrows(arrows), m_boxes(boxes) {}

  virtual bool keyDown(int key) {
    typedef osgGA::GUIEventAdapter::KeySymbol osgKey;
    switch (key) {
      case osgKey::KEY_F1:
        for (size_t i = 0; i < m_arrows->size(); i++)
          m_arrows->at(i)->setNodeMask(~m_arrows->at(i)->getNodeMask());
        return true;
      case osgKey::KEY_F2:
        for (size_t i = 0; i < m_boxes->size(); i++)
          m_boxes->at(i)->setNodeMask(~m_boxes->at(i)->getNodeMask());
        return true;
      default:
        return false;
    }
  }
};

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  float radius = 12.7;
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
  viewer.addEventHandler(
      new MyKeyboardHandler(sphere.m_arrows, sphere.m_boxes));
  return viewer.run();
}
