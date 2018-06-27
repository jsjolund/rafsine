#pragma once
#include "Voxel.hpp"
#include "ColorSet.hpp"

//This class can build and display a mesh based on an voxel array and a color set
class VoxelMesh
{
private:
  //voxels to base the mesh on
  //TODO: shared pointer from the ressource manager
  VoxelArray *voxels_;
  //color set used for this mesh
  //TODO: shared pointer from the ressource manager
  ColorSet *colors_;
  //boolean wich states if the mesh has been generated
  bool mesh_ready_;
  //vertices from the generated mesh
  std::vector<vec3r> vertices_;
  //color of each vertex
  std::vector<vec3r> v_colors_;
  //position of the mesh in the world
  vec3r position_;
  // orientation of the mesh
  pol3 orientation_;
  // size of the voxels (1 == default size)
  real size_;
  //Compute a simple local ambient occlusion
  void computeSimpleAO(vec3ui position, vec3ui normal, vec3ui perp1, vec3ui perp2,
                       real &AO1, real &AO2, real &AO3, real &AO4);
  //values to shadowing each face
  real shadowXpos, shadowXneg, shadowYpos, shadowYneg, shadowZpos, shadowZneg;
  //enable the ambient occlusion
  bool AO_enabled_;

public:
  ///Constructor from a file on the disk
  /// TODO: to be modified with the ressource manager
  VoxelMesh(std::string voxel_file_name, vec3r position = vec3r::ZERO, vec3r orientation = vec3r::X, real size = 1);
  /// Constructor with an existing voxel array
  VoxelMesh(const VoxelArray &voxels, vec3r position = vec3r::ZERO, vec3r orientation = vec3r::X, real size = 1);
  ///Copy constructor
  VoxelMesh(const VoxelMesh &voxmesh);
  //assignment operator
  VoxelMesh &operator=(const VoxelMesh &voxmesh);
  /// Destructor
  ~VoxelMesh()
  {
    //TODO: to be remove after the Ressource Manager
    delete voxels_;
    delete colors_;
  }

  //Basic set and get functions
  //TODO: need to be inline?
  inline void setPosition(vec3r position) { position_ = position; }
  inline vec3r getPosition() const { return position_; }
  inline void setOrientation(pol3 orientation) { orientation_ = orientation; }
  inline void setOrientation(vec3r orientation) { orientation_ = orientation; }
  inline pol3 getOrientation() const { return orientation_; }
  inline void setSize(real size) { size_ = size; }
  inline real getSize() { return size_; }
  //compute an aproximate radius from size_ and voxels_
  inline real getRadius()
  {
    int nx = voxels_->getSizeX();
    int ny = voxels_->getSizeY();
    return size_ * sqrt(nx * nx + ny * ny);
  }

  //return the voxel pointer to the voxel array
  inline VoxelArray *voxels() const { return voxels_; }
  //return a reference to the voxel (i,j,k)
  inline voxel &voxels(unsigned int i, unsigned int j, unsigned int k) const { return (*voxels_)(i, j, k); }
  //use is empty function from the voxel array class
  inline bool isEmpty(unsigned int i, unsigned int j, unsigned int k) const { return voxels_->isEmpty(i, j, k); }

  ///Build the mesh
  void buildMesh(float xmin = -1,
                 float xmax = -1,
                 float ymin = -1,
                 float ymax = -1,
                 float zmin = -1,
                 float zmax = -1);
  //display the object without any translation, rotation, or scaling
  void displayNoTransform(); // const;
  //display the object
  void display(); // const;
  inline void enableAO() { AO_enabled_ = true; }
  inline void disableAO() { AO_enabled_ = false; }
  inline void disableShading()
  {
    disableAO();
    shadowXpos = shadowXneg = shadowYpos = shadowYneg = shadowZpos = shadowZneg = 1;
  }
};
