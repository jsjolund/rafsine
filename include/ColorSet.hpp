#pragma once
#include <stdlib.h>
#include <utility>
#include <map>

#include <glm/vec3.hpp>

#define MAX_COLORSET_SIZE 256

/// A colorKey is a couple of a number and a color
/** This structure is used by the VoxelVisu class to construct color set */
typedef std::pair<int, glm::vec3> colorKey;

//Defines a color set for voxels
class ColorSet
{
private:
  typedef std::map<int, glm::vec3> ColorMap;
  //define the color set
  ColorMap m_colorSet;

public:
  //Constructor
  ColorSet()
  {
    //load the default colors
    loadDefault();
  }
  //Load the default colors
  void loadDefault()
  {
    m_colorSet.clear();
    m_colorSet[0] = glm::vec3(1., 1., 1.);
    m_colorSet[1] = glm::vec3(1., 1., 1.);
    srand(3);
    for (int i = 2; i < MAX_COLORSET_SIZE; i++)
    {
      m_colorSet[i] = glm::vec3((1.0 / RAND_MAX) * rand(), (1.0 / RAND_MAX) * rand(), (1.0 / RAND_MAX) * rand());
      //m_colorSet[i] = col3( 1*i/68.0, 1*i/68.0, 1*i/68.0 );
    }
  }
  /// Empty the color set (in order to create a new one)
  void clear()
  {
    m_colorSet.clear();
  }
  /// Add a color key to the color set
  void addColorKey(colorKey key) { m_colorSet[key.first] = key.second; }
  /// get the color of the key
  glm::vec3 getColor(int key) const { return m_colorSet.find(key)->second; }
  /// return the number of different colors in the color set
  int getSize() const { return m_colorSet.size(); }
};
