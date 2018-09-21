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
#include "BoundaryCondition.hpp"

/// Store a 3D array of voxels
class VoxelArray
{
private:
  // size of the domain
  unsigned int m_sizeX, m_sizeY, m_sizeZ;
  // pointer to the data on the cpu
  voxel *m_data;
  // data on the GPU
  voxel *m_data_d;

public:
  //constructor
  VoxelArray(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ)
      : m_sizeX(sizeX), m_sizeY(sizeY), m_sizeZ(sizeZ)
  {
    m_data = new voxel[getFullSize()];
    memset(m_data, 0, getFullSize() * sizeof(voxel));
    cudaMalloc((void **)&m_data_d, sizeof(voxel) * getFullSize());
  }
  //Copy constructor
  VoxelArray(const VoxelArray &other)
      : m_sizeX(other.m_sizeX), m_sizeY(other.m_sizeY), m_sizeZ(other.m_sizeZ)
  {
    m_data = new voxel[other.getFullSize()];
    memcpy(m_data, other.m_data, sizeof(voxel) * getFullSize());
    cudaMalloc((void **)&m_data_d, sizeof(voxel) * getFullSize());
  }
  //Assignment operator
  VoxelArray &operator=(const VoxelArray &other);

  //destructor
  ~VoxelArray()
  {
    delete[] m_data;
  }
  inline unsigned int getSizeX() const { return m_sizeX; }
  inline unsigned int getSizeY() const { return m_sizeY; }
  inline unsigned int getSizeZ() const { return m_sizeZ; }
  inline unsigned int getFullSize() const { return m_sizeX * m_sizeY * m_sizeZ; }
  //provide easy read and write access to voxels
  inline voxel &operator()(unsigned int x, unsigned int y, unsigned int z)
  {
    return m_data[x + y * m_sizeX + z * m_sizeX * m_sizeY];
  }
  //provide read-only access
  inline voxel getVoxelReadOnly(unsigned int x, unsigned int y, unsigned int z) const
  {
    return m_data[x + y * m_sizeX + z * m_sizeX * m_sizeY];
  }
  //send the data to the GPU
  inline void upload()
  {
    cudaMemcpy(m_data_d, m_data, sizeof(voxel) * getFullSize(), cudaMemcpyHostToDevice);
  }
  //returns a pointer to the gpu data
  inline voxel *gpu_ptr() { return m_data_d; }
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
      m_data[i] = value;
  }
};

#endif
