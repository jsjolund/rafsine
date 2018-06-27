/**************************************************************************/ /**
 * \file Voxel.hpp
 * \brief Define an array of voxel class
 * \author Nicolas Delbosc
 * \date June 2014
 *****************************************************************************/
#pragma once
#ifndef VOXEL_HPP
#define VOXEL_HPP

#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <string.h>

#include "Primitives.hpp"

/// The voxel data are stored as unsigned char, so each voxel takes 1 byte of memory
typedef int voxel;

/// Store a 3D array of voxels
class VoxelArray
{
private:
  // size of the domain
  unsigned int sizeX_, sizeY_, sizeZ_;
  // pointer to the data on the cpu
  voxel *data_;
  // data on the GPU
  voxel *data_d;

public:
  //type of empty voxels
  static const voxel VOX_EMPTY = -1;
  //type of fluid voxels
  static const voxel VOX_FLUID = 0;
  //constructor
  VoxelArray(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ)
      : sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ)
  {
    data_ = new voxel[getFullSize()];
    cudaMalloc((void **)&data_d, sizeof(voxel) * getFullSize());
  }
  //Copy constructor
  VoxelArray(const VoxelArray &other)
      : sizeX_(other.sizeX_), sizeY_(other.sizeY_), sizeZ_(other.sizeZ_)
  {
    data_ = new voxel[other.getFullSize()];
    memcpy(data_, other.data_, sizeof(voxel) * getFullSize());
    cudaMalloc((void **)&data_d, sizeof(voxel) * getFullSize());
  }
  //Assignment operator
  VoxelArray &operator=(const VoxelArray &other);

  //destructor
  ~VoxelArray()
  {
    delete[] data_;
  }
  inline unsigned int getSizeX() const { return sizeX_; }
  inline unsigned int getSizeY() const { return sizeY_; }
  inline unsigned int getSizeZ() const { return sizeZ_; }
  inline unsigned int getFullSize() const { return sizeX_ * sizeY_ * sizeZ_; }
  //provide easy read and write access to voxels
  inline voxel &operator()(unsigned int x, unsigned int y, unsigned int z)
  {
    return data_[x + y * sizeX_ + z * sizeX_ * sizeY_];
  }
  //provide read-only access
  inline voxel getVoxelReadOnly(unsigned int x, unsigned int y, unsigned int z) const
  {
    return data_[x + y * sizeX_ + z * sizeX_ * sizeY_];
  }
  //send the data to the GPU
  inline void upload()
  {
    cudaMemcpy(data_d, data_, sizeof(voxel) * getFullSize(), cudaMemcpyHostToDevice);
  }
  //returns a pointer to the gpu data
  inline voxel *gpu_ptr() { return data_d; }
  // return true if the voxel is of type VOX_EMPTY or VOX_FLUID
  // check that the coordinates are inside the domain
  /// \TODO if unsigned int no need for <0 case and no need for tx,ty,tz
  //  \TODO fluid is empty
  bool isEmpty(unsigned int x, unsigned int y, unsigned int z) const;

  //fluid is not empty
  //tody isEmpty or isTransparent
  bool isEmptyStrict(unsigned int x, unsigned int y, unsigned int z) const;

  inline bool isEmpty(vec3ui position) const
  {
    return isEmpty(position.x, position.y, position.z);
  }
  //transform the empty voxels (-1) into the last type (39) so they are visible
  //TODO inline void showEmptyVoxels()
  //function to save the voxels
  void saveToFile(std::string filename);

  //function to save only the chunk of voxels that are not empty
  void saveAutocrop(std::string filename);

  /// Function to load the voxels from a file
  void loadFromFile(std::string filename);

  /// Fill the whole array with a unique value
  inline void fill(voxel value)
  {
    for (unsigned int i = 0; i < getFullSize(); i++)
      data_[i] = value;
  }
};

#endif
