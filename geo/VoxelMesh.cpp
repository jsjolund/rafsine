#include "VoxelMesh.hpp"

///Constructor from a file on the disk
/// TODO: to be modified with the ressource manager
VoxelMesh::VoxelMesh(std::string voxel_file_name, vec3r position, vec3r orientation, real size)
    : mesh_ready_(false),
      position_(position),
      orientation_(orientation),
      size_(size),
      shadowXpos(0.8), shadowXneg(0.4), shadowYpos(0.7), shadowYneg(0.5), shadowZpos(1.0), shadowZneg(0.3),
      AO_enabled_(false)
{
  //load file size
  std::ifstream fin(voxel_file_name.c_str());
  unsigned int nx, ny, nz;
  fin >> nx >> ny >> nz;
  fin.close();
  voxels_ = new VoxelArray(nx, ny, nz);
  voxels_->loadFromFile(voxel_file_name);
  //create the color set
  colors_ = new ColorSet();

  vertices_ = new osg::Vec3Array;
  v_colors_ = new osg::Vec4Array;
  normals_ = new osg::Vec3Array;
}

/// Constructor with an existing voxel array
VoxelMesh::VoxelMesh(const VoxelArray &voxels, vec3r position, vec3r orientation, real size)
    : mesh_ready_(false),
      position_(position),
      orientation_(orientation),
      size_(size),
      shadowXpos(0.8), shadowXneg(0.4), shadowYpos(0.7), shadowYneg(0.5), shadowZpos(1.0), shadowZneg(0.3),
      AO_enabled_(false)
{
  //Use the existing voxelArray
  voxels_ = new VoxelArray(voxels);
  //create the color set
  colors_ = new ColorSet();

  vertices_ = new osg::Vec3Array;
  v_colors_ = new osg::Vec4Array;
  normals_ = new osg::Vec3Array;
}

///Copy constructor
VoxelMesh::VoxelMesh(const VoxelMesh &voxmesh)
    : mesh_ready_(voxmesh.mesh_ready_),
      vertices_(voxmesh.vertices_),
      v_colors_(voxmesh.v_colors_),
      normals_(voxmesh.normals_),
      position_(voxmesh.position_),
      orientation_(voxmesh.orientation_),
      size_(voxmesh.size_),
      shadowXpos(voxmesh.shadowXpos),
      shadowXneg(voxmesh.shadowXneg),
      shadowYpos(voxmesh.shadowYpos),
      shadowYneg(voxmesh.shadowYneg),
      shadowZpos(voxmesh.shadowZpos),
      shadowZneg(voxmesh.shadowZneg),
      AO_enabled_(voxmesh.AO_enabled_)
{
  voxels_ = new VoxelArray(*voxmesh.voxels_);
  colors_ = new ColorSet(*voxmesh.colors_);
}

//assignment operator
VoxelMesh &VoxelMesh::operator=(const VoxelMesh &voxmesh)
{
  voxels_ = voxmesh.voxels_;
  colors_ = voxmesh.colors_;
  normals_ = voxmesh.normals_;
  mesh_ready_ = mesh_ready_;
  vertices_ = voxmesh.vertices_;
  v_colors_ = voxmesh.v_colors_;
  position_ = voxmesh.position_;
  orientation_ = voxmesh.orientation_;
  size_ = voxmesh.size_;
  AO_enabled_ = voxmesh.AO_enabled_;
  return *this;
}

//Compute a simple local ambient occlusion
void VoxelMesh::computeSimpleAO(vec3ui position, vec3ui normal, vec3ui perp1, vec3ui perp2,
                                real &AO1, real &AO2, real &AO3, real &AO4)
{
  if (!AO_enabled_)
  {
    AO1 = AO2 = AO3 = AO4 = 1;
  }
  else
  {
    bool yp = voxels_->isEmpty(position + normal + perp1);
    bool yn = voxels_->isEmpty(position + normal - perp1);
    bool zp = voxels_->isEmpty(position + normal + perp2);
    bool zn = voxels_->isEmpty(position + normal - perp2);
    bool ypzp = voxels_->isEmpty(position + normal + perp1 + perp2);
    bool ypzn = voxels_->isEmpty(position + normal + perp1 - perp2);
    bool ynzp = voxels_->isEmpty(position + normal - perp1 + perp2);
    bool ynzn = voxels_->isEmpty(position + normal - perp1 - perp2);
    AO1 = 0.75f * (yn ^ zn) + (yn & zn) - 0.25f * (!ynzn) * (yn | zn);
    AO2 = 0.75f * (yp ^ zn) + (yp & zn) - 0.25f * (!ypzn) * (yp | zn);
    AO3 = 0.75f * (yp ^ zp) + (yp & zp) - 0.25f * (!ypzp) * (yp | zp);
    AO4 = 0.75f * (yn ^ zp) + (yn & zp) - 0.25f * (!ynzp) * (yn | zp);
  }
}

//build the mesh for the voxel array
void VoxelMesh::buildMesh(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
{
  //cout << "Voxel Mesh : buildMesh(), size=("<<voxels_->getSizeX()<<","<<voxels_->getSizeY()<<","<<voxels_->getSizeZ()<<")" << endl;
  //cout << "Voxel Mesh : buildMesh(), min and max =("<<xmin<<","<<xmax<<","<<ymin<<","<<ymax<<","<<zmin<<","<<zmax<<")" << endl;
  mesh_ready_ = true;
  //Important: reset any previous mesh
  vertices_->clear();
  v_colors_->clear();
  normals_->clear();
  for (int k = 0; k < int(voxels_->getSizeZ()); ++k)
  {
    //cout << "\rBuild Mesh : " << int(100*k/real(voxels_->getSizeZ())) << " %           ";
    //fflush(stdout);
    for (int j = 0; j < int(voxels_->getSizeY()); ++j)
      for (int i = 0; i < int(voxels_->getSizeX()); ++i)
      {
        if (zmax >= 0) // if the croping is in use
        {
          if ((i < xmin) || (j < ymin) || (k < zmin) || (i > xmax) || (j > ymax) || (k > zmax))
            continue;
        }
        if (!voxels_->isEmpty(i, j, k))
        {
          voxel v = voxels_->getVoxelReadOnly(i, j, k);
          col3 col_col3 = colors_->getColor(v);
          osg::Vec4 col_vec3r(col_col3.r / 255., col_col3.g / 255., col_col3.b / 255., 1.0);
          osg::Vec4 shad_col = col_vec3r;

          real AO1, AO2, AO3, AO4;
          if (voxels_->isEmpty(i + 1, j, k))
          {
            vertices_->push_back(osg::Vec3(i + 1, j, k));
            vertices_->push_back(osg::Vec3(i + 1, j + 1, k));
            vertices_->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            vertices_->push_back(osg::Vec3(i + 1, j, k + 1));
            //v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r);
            computeSimpleAO(vec3ui(i, j, k), vec3ui::X, vec3ui::Y, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowXpos;
            v_colors_->push_back(shad_col * AO1);
            v_colors_->push_back(shad_col * AO2);
            v_colors_->push_back(shad_col * AO3);
            v_colors_->push_back(shad_col * AO4);
            osg::Vec3 normal(1.0f, 0.0f, 0.0f);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
          }
          if (voxels_->isEmpty(i - 1, j, k))
          {
            vertices_->push_back(osg::Vec3(i, j, k));
            vertices_->push_back(osg::Vec3(i, j + 1, k));
            vertices_->push_back(osg::Vec3(i, j + 1, k + 1));
            vertices_->push_back(osg::Vec3(i, j, k + 1));
            //v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r);
            computeSimpleAO(vec3ui(i, j, k), -vec3ui::X, vec3ui::Y, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowXneg;
            v_colors_->push_back(shad_col * AO1);
            v_colors_->push_back(shad_col * AO2);
            v_colors_->push_back(shad_col * AO3);
            v_colors_->push_back(shad_col * AO4);
            osg::Vec3 normal(-1.0f, 0.0f, 0.0f);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
          }
          if (voxels_->isEmpty(i, j + 1, k))
          {
            vertices_->push_back(osg::Vec3(i, j + 1, k));
            vertices_->push_back(osg::Vec3(i + 1, j + 1, k));
            vertices_->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            vertices_->push_back(osg::Vec3(i, j + 1, k + 1));
            //v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r);
            computeSimpleAO(vec3ui(i, j, k), vec3ui::Y, vec3ui::X, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowYpos;
            v_colors_->push_back(shad_col * AO1);
            v_colors_->push_back(shad_col * AO2);
            v_colors_->push_back(shad_col * AO3);
            v_colors_->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, 1.0f, 0.0f);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
          }
          if (voxels_->isEmpty(i, j - 1, k))
          {
            vertices_->push_back(osg::Vec3(i, j, k));
            vertices_->push_back(osg::Vec3(i + 1, j, k));
            vertices_->push_back(osg::Vec3(i + 1, j, k + 1));
            vertices_->push_back(osg::Vec3(i, j, k + 1));
            //v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r);
            computeSimpleAO(vec3ui(i, j, k), -vec3ui::Y, vec3ui::X, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowYneg;
            v_colors_->push_back(shad_col * AO1);
            v_colors_->push_back(shad_col * AO2);
            v_colors_->push_back(shad_col * AO3);
            v_colors_->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, -1.0f, 0.0f);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
          }
          if (voxels_->isEmpty(i, j, k + 1))
          {
            vertices_->push_back(osg::Vec3(i, j, k + 1));
            vertices_->push_back(osg::Vec3(i + 1, j, k + 1));
            vertices_->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            vertices_->push_back(osg::Vec3(i, j + 1, k + 1));
            //v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r);
            computeSimpleAO(vec3ui(i, j, k), vec3ui::Z, vec3ui::X, vec3ui::Y, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowZpos;
            v_colors_->push_back(shad_col * AO1);
            v_colors_->push_back(shad_col * AO2);
            v_colors_->push_back(shad_col * AO3);
            v_colors_->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, 0.0f, 1.0f);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
          }
          if (voxels_->isEmpty(i, j, k - 1))
          {
            vertices_->push_back(osg::Vec3(i, j, k));
            vertices_->push_back(osg::Vec3(i + 1, j, k));
            vertices_->push_back(osg::Vec3(i + 1, j + 1, k));
            vertices_->push_back(osg::Vec3(i, j + 1, k));
            //v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r); v_colors_->push_back(col_vec3r);
            computeSimpleAO(vec3ui(i, j, k), -vec3ui::Z, vec3ui::X, vec3ui::Y, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowZneg;
            v_colors_->push_back(shad_col * AO1);
            v_colors_->push_back(shad_col * AO2);
            v_colors_->push_back(shad_col * AO3);
            v_colors_->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, 0.0f, -1.0f);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
            normals_->push_back(normal);
          }
        }
      }
  }
  vertices_->trim();
  v_colors_->trim();
}
