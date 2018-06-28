#pragma once
#include <stdlib.h>
#include "Primitives.hpp"

/// A colorKey is a couple of a number and a color
/** This structure is used by the VoxelVisu class to construct color set */
typedef pair_<unsigned char, col3> colorKey;

//Defines a color set for voxels
class ColorSet
{
private:
  typedef std::map<unsigned char, col3> ColorMap;
  //define the color set
  ColorMap color_set_;

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
    color_set_.clear();
    color_set_[0] = col3(255., 255., 255.);
    color_set_[1] = col3(255., 255., 255.);
    srand(3);
    for (int i = 2; i < 256; i++)
    {
      color_set_[i] = col3((255.0 / RAND_MAX) * rand(), (255.0 / RAND_MAX) * rand(), (255.0 / RAND_MAX) * rand());
      //color_set_[i] = col3( 255*i/68.0, 255*i/68.0, 255*i/68.0 );
    }
  }
  /// Empty the color set (in order to create a new one)
  void clear()
  {
    color_set_.clear();
  }
  /// Add a color key to the color set
  void addColorKey(colorKey key) { color_set_[key.key] = key.value; }
  /// get the color of the key
  col3 getColor(voxel key) const { return color_set_.find(key)->second; }
  /// return the number of different colors in the color set
  int getSize() const { return color_set_.size(); }
};