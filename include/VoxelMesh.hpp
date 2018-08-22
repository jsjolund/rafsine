#pragma once

#include <osg/Vec3>
#include <osg/Array>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/Geode>
#include <OpenThreads/Thread>

#include "Voxel.hpp"
#include "ColorSet.hpp"

//This class can build and display a mesh based on an voxel array and a color set
class VoxelMesh : public osg::Geometry
{
private:
  // Voxels to base the mesh on
  VoxelArray *m_voxels;
  // Color set used for this mesh
  ColorSet *m_colorSet;
  // Size of the voxels (1 == default size)
  real m_size;

  //Compute a simple local ambient occlusion
  void computeSimpleAO(vec3ui position, vec3ui normal, vec3ui perp1, vec3ui perp2,
                       real &AO1, real &AO2, real &AO3, real &AO4);

  // Values to shadowing each face
  real shadowXpos, shadowXneg, shadowYpos, shadowYneg, shadowZpos, shadowZneg;
  // Enable the ambient occlusion
  bool m_AOenabled;

  // World transform
  osg::ref_ptr<osg::PositionAttitudeTransform> m_transform;

protected:
  ~VoxelMesh()
  {
    delete m_colorSet;
  }

public:
  // Vertices from the generated mesh
  osg::ref_ptr<osg::Vec3Array> m_vertexArray;
  // Color of each vertex
  osg::ref_ptr<osg::Vec4Array> m_colorArray;
  // Plane normals
  osg::ref_ptr<osg::Vec3Array> m_normalsArray;

  inline osg::ref_ptr<osg::PositionAttitudeTransform> getTransform() { return m_transform; }

  // Constructor from a file on the disk
  VoxelMesh(std::string voxel_file_name, real size = 1);
  // Constructor with an existing voxel array
  VoxelMesh(const VoxelArray &voxels, real size = 1);
  // Copy constructor
  VoxelMesh(const VoxelMesh &voxmesh);
  // Assignment operator
  VoxelMesh &operator=(const VoxelMesh &voxmesh);
  // Destructor

  // Basic set and get functions
  inline void setSize(real size) { m_size = size; }
  inline real getSize() { return m_size; }
  inline int getSizeX() { return m_voxels->getSizeX(); }
  inline int getSizeY() { return m_voxels->getSizeY(); }
  inline int getSizeZ() { return m_voxels->getSizeZ(); }

  // Compute an aproximate radius from m_size and m_voxels
  inline real getRadius()
  {
    int nx = m_voxels->getSizeX();
    int ny = m_voxels->getSizeY();
    return m_size * sqrt(nx * nx + ny * ny);
  }

  // Return the voxel pointer to the voxel array
  inline VoxelArray *voxels() const { return m_voxels; }
  // Return a reference to the voxel (i,j,k)
  inline voxel &voxels(unsigned int i, unsigned int j, unsigned int k) const { return (*m_voxels)(i, j, k); }
  // Use is empty function from the voxel array class
  inline bool isEmpty(unsigned int i, unsigned int j, unsigned int k) const { return m_voxels->isEmpty(i, j, k); }

  // Build the mesh
  void buildMesh(osg::Vec3i m_voxMin, osg::Vec3i voxMax);
  // Display the object without any translation, rotation, or scaling
  inline void enableAO() { m_AOenabled = true; }
  inline void disableAO() { m_AOenabled = false; }
  inline void disableShading()
  {
    disableAO();
    shadowXpos = shadowXneg = shadowYpos = shadowYneg = shadowZpos = shadowZneg = 1;
  }
  
};
