#pragma once

#include <osg/Vec4>

#include <stdlib.h>
#include <map>
#include <utility>

#define MAX_COLORSET_SIZE 2048

/**
 * @brief A colorKey is a couple of a number and a color
 */
typedef std::pair<int, osg::Vec4> colorKey;

/**
 * @brief Defines a color set for voxels
 */
class ColorSet {
 private:
  typedef std::map<int, osg::Vec4> ColorMap;
  ColorMap m_colorSet;

 public:
  /**
   * @brief Get the color of the index key
   *
   * @param key Index
   * @return osg::Vec4 The color vector
   */
  osg::Vec4 getColor(int key) const { return m_colorSet.find(key)->second; }

  /**
   * @return Number of different colors in the color set
   */
  int getSize() const { return m_colorSet.size(); }

  /**
   * @brief Get a random color
   *
   * @param int Pointer to random seed
   * @return osg::Vec4 The color vector
   */
  osg::Vec4 getRandomColor(unsigned int* seed) {
    return osg::Vec4((1.0 / RAND_MAX) * rand_r(seed),
                     (1.0 / RAND_MAX) * rand_r(seed),
                     (1.0 / RAND_MAX) * rand_r(seed), 1.0);
  }

  ColorSet() {
    // Default colors
    m_colorSet.clear();
    m_colorSet[0] = osg::Vec4(1., 1., 1., 1.);
    unsigned int seed = 4;
    for (int i = 1; i < MAX_COLORSET_SIZE; i++) {
      m_colorSet[i] = getRandomColor(&seed);
    }
  }

  ColorSet& operator=(const ColorSet& other) {
    m_colorSet = other.m_colorSet;
    return *this;
  }
};
