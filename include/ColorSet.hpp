#pragma once

#include <osg/Vec4>

#include <stdlib.h>
#include <map>
#include <utility>

#define MAX_COLORSET_SIZE 256

/**
 * @brief A colorKey is a couple of a number and a color
 *
 */
typedef std::pair<int, osg::Vec4> colorKey;

/**
 * @brief Defines a color set for voxels
 *
 */
class ColorSet {
 private:
  typedef std::map<int, osg::Vec4> ColorMap;
  // define the color set
  ColorMap m_colorSet;

 public:
  // Constructor
  ColorSet() {
    // load the default colors
    m_colorSet.clear();
    m_colorSet[0] = osg::Vec4(1., 1., 1., 1.);
    m_colorSet[1] = osg::Vec4(1., 1., 1., 1.);
    srand(3);
    for (int i = 2; i < MAX_COLORSET_SIZE; i++) {
      m_colorSet[i] =
          osg::Vec4((1.0 / RAND_MAX) * rand(), (1.0 / RAND_MAX) * rand(),
                    (1.0 / RAND_MAX) * rand(), 1.0);
    }
  }
  /// Empty the color set (in order to create a new one)
  void clear() { m_colorSet.clear(); }
  /// Add a color key to the color set
  void addColorKey(colorKey key) { m_colorSet[key.first] = key.second; }
  /// get the color of the key
  osg::Vec4 getColor(int key) const { return m_colorSet.find(key)->second; }
  /// return the number of different colors in the color set
  int getSize() const { return m_colorSet.size(); }
};
