#pragma once
#include <stdlib.h>
#include "Primitives.hpp"

#define MAX_COLORSET_SIZE 256

/// A colorKey is a couple of a number and a color
/** This structure is used by the VoxelVisu class to construct color set */
typedef pair_<unsigned char, col3> colorKey;

//Defines a color set for voxels
class ColorSet
{
private:
  typedef std::map<unsigned char, col3> ColorMap;
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
    m_colorSet[0] = col3(255., 255., 255.);
    m_colorSet[1] = col3(255., 255., 255.);
    srand(3);
    for (int i = 2; i < MAX_COLORSET_SIZE; i++)
    {
      m_colorSet[i] = col3((255.0 / RAND_MAX) * rand(), (255.0 / RAND_MAX) * rand(), (255.0 / RAND_MAX) * rand());
      //m_colorSet[i] = col3( 255*i/68.0, 255*i/68.0, 255*i/68.0 );
    }
  }
  /// Empty the color set (in order to create a new one)
  void clear()
  {
    m_colorSet.clear();
  }
  /// Add a color key to the color set
  void addColorKey(colorKey key) { m_colorSet[key.key] = key.value; }
  /// get the color of the key
  col3 getColor(voxel key) const { return m_colorSet.find(key)->second; }
  /// return the number of different colors in the color set
  int getSize() const { return m_colorSet.size(); }
};
