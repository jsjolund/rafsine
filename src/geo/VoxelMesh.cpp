#include "VoxelMesh.hpp"

// Constructor from a file on the disk
// TODO: to be modified with the ressource manager
VoxelMesh::VoxelMesh(std::string voxel_file_name, real size)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_size(size),
      shadowXpos(0.8), shadowXneg(0.4),
      shadowYpos(0.7), shadowYneg(0.5),
      shadowZpos(1.0), shadowZneg(0.3),
      m_AOenabled(false)
{
  //load file size
  std::ifstream fin(voxel_file_name.c_str());
  unsigned int nx, ny, nz;
  fin >> nx >> ny >> nz;
  fin.close();

  m_voxels = new VoxelArray(nx, ny, nz);
  m_voxels->loadFromFile(voxel_file_name);

  m_colorSet = new ColorSet();
  m_vertexArray = new osg::Vec3Array();
  m_colorArray = new osg::Vec4Array();
  m_normalsArray = new osg::Vec3Array();
  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));
}

// Constructor with an existing voxel array
VoxelMesh::VoxelMesh(const VoxelArray &voxels, real size)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_size(size),
      shadowXpos(0.8), shadowXneg(0.4),
      shadowYpos(0.7), shadowYneg(0.5),
      shadowZpos(1.0), shadowZneg(0.3),
      m_AOenabled(false)
{
  m_voxels = new VoxelArray(voxels);
  m_colorSet = new ColorSet();
  m_vertexArray = new osg::Vec3Array();
  m_colorArray = new osg::Vec4Array();
  m_normalsArray = new osg::Vec3Array();
  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));
}

// Copy constructor
VoxelMesh::VoxelMesh(const VoxelMesh &voxmesh)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_vertexArray(voxmesh.m_vertexArray),
      m_colorArray(voxmesh.m_colorArray),
      m_normalsArray(voxmesh.m_normalsArray),
      m_size(voxmesh.m_size),
      shadowXpos(voxmesh.shadowXpos),
      shadowXneg(voxmesh.shadowXneg),
      shadowYpos(voxmesh.shadowYpos),
      shadowYneg(voxmesh.shadowYneg),
      shadowZpos(voxmesh.shadowZpos),
      shadowZneg(voxmesh.shadowZneg),
      m_AOenabled(voxmesh.m_AOenabled)
{
  m_voxels = new VoxelArray(*voxmesh.m_voxels);
  m_colorSet = new ColorSet(*voxmesh.m_colorSet);
  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));
}

//assignment operator
VoxelMesh &VoxelMesh::operator=(const VoxelMesh &voxmesh)
{
  m_voxels = voxmesh.m_voxels;
  m_colorSet = voxmesh.m_colorSet;
  m_normalsArray = voxmesh.m_normalsArray;
  m_vertexArray = voxmesh.m_vertexArray;
  m_colorArray = voxmesh.m_colorArray;
  m_size = voxmesh.m_size;
  m_AOenabled = voxmesh.m_AOenabled;
  return *this;
}

//Compute a simple local ambient occlusion
void VoxelMesh::computeSimpleAO(vec3ui position, vec3ui normal, vec3ui perp1, vec3ui perp2,
                                real &AO1, real &AO2, real &AO3, real &AO4)
{
  if (!m_AOenabled)
  {
    AO1 = AO2 = AO3 = AO4 = 1;
  }
  else
  {
    bool yp = m_voxels->isEmpty(position + normal + perp1);
    bool yn = m_voxels->isEmpty(position + normal - perp1);
    bool zp = m_voxels->isEmpty(position + normal + perp2);
    bool zn = m_voxels->isEmpty(position + normal - perp2);
    bool ypzp = m_voxels->isEmpty(position + normal + perp1 + perp2);
    bool ypzn = m_voxels->isEmpty(position + normal + perp1 - perp2);
    bool ynzp = m_voxels->isEmpty(position + normal - perp1 + perp2);
    bool ynzn = m_voxels->isEmpty(position + normal - perp1 - perp2);
    AO1 = 0.75f * (yn ^ zn) + (yn & zn) - 0.25f * (!ynzn) * (yn | zn);
    AO2 = 0.75f * (yp ^ zn) + (yp & zn) - 0.25f * (!ypzn) * (yp | zn);
    AO3 = 0.75f * (yp ^ zp) + (yp & zp) - 0.25f * (!ypzp) * (yp | zp);
    AO4 = 0.75f * (yn ^ zp) + (yn & zp) - 0.25f * (!ynzp) * (yn | zp);
  }
}

//build the mesh for the voxel array
void VoxelMesh::buildMesh(osg::Vec3i voxMin, osg::Vec3i voxMax)
{
  m_vertexArray->clear();
  m_colorArray->clear();
  m_normalsArray->clear();

  m_vertexArray->trim();
  m_colorArray->trim();
  m_normalsArray->trim();

  for (int k = 0; k < int(m_voxels->getSizeZ()); ++k)
    for (int j = 0; j < int(m_voxels->getSizeY()); ++j)
      for (int i = 0; i < int(m_voxels->getSizeX()); ++i)
      {
        if (voxMax.z() >= 0) // if the croping is in use
        {
          if (i < voxMin.x() || (j < voxMin.y()) || (k < voxMin.z()) || (i > voxMax.x()) || (j > voxMax.y()) || (k > voxMax.z()))
            continue;
        }
        if (!m_voxels->isEmpty(i, j, k))
        {
          voxel v = m_voxels->getVoxelReadOnly(i, j, k);
          col3 col_col3 = m_colorSet->getColor(v);
          osg::Vec4 col_vec3r(col_col3.r / 255., col_col3.g / 255., col_col3.b / 255., 1.0);
          osg::Vec4 shad_col = col_vec3r;

          real AO1, AO2, AO3, AO4;
          if (m_voxels->isEmpty(i + 1, j, k))
          {
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            computeSimpleAO(vec3ui(i, j, k), vec3ui::X, vec3ui::Y, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowXpos;
            m_colorArray->push_back(shad_col * AO1);
            m_colorArray->push_back(shad_col * AO2);
            m_colorArray->push_back(shad_col * AO3);
            m_colorArray->push_back(shad_col * AO4);
            osg::Vec3 normal(1.0f, 0.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i - 1, j, k))
          {
            m_vertexArray->push_back(osg::Vec3(i, j, k));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j, k + 1));
            computeSimpleAO(vec3ui(i, j, k), -vec3ui::X, vec3ui::Y, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowXneg;
            m_colorArray->push_back(shad_col * AO1);
            m_colorArray->push_back(shad_col * AO2);
            m_colorArray->push_back(shad_col * AO3);
            m_colorArray->push_back(shad_col * AO4);
            osg::Vec3 normal(-1.0f, 0.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j + 1, k))
          {
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k + 1));
            computeSimpleAO(vec3ui(i, j, k), vec3ui::Y, vec3ui::X, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowYpos;
            m_colorArray->push_back(shad_col * AO1);
            m_colorArray->push_back(shad_col * AO2);
            m_colorArray->push_back(shad_col * AO3);
            m_colorArray->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, 1.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j - 1, k))
          {
            m_vertexArray->push_back(osg::Vec3(i, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j, k + 1));
            computeSimpleAO(vec3ui(i, j, k), -vec3ui::Y, vec3ui::X, vec3ui::Z, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowYneg;
            m_colorArray->push_back(shad_col * AO1);
            m_colorArray->push_back(shad_col * AO2);
            m_colorArray->push_back(shad_col * AO3);
            m_colorArray->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, -1.0f, 0.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j, k + 1))
          {
            m_vertexArray->push_back(osg::Vec3(i, j, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k + 1));
            computeSimpleAO(vec3ui(i, j, k), vec3ui::Z, vec3ui::X, vec3ui::Y, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowZpos;
            m_colorArray->push_back(shad_col * AO1);
            m_colorArray->push_back(shad_col * AO2);
            m_colorArray->push_back(shad_col * AO3);
            m_colorArray->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, 0.0f, 1.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
          if (m_voxels->isEmpty(i, j, k - 1))
          {
            m_vertexArray->push_back(osg::Vec3(i, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i, j + 1, k));
            computeSimpleAO(vec3ui(i, j, k), -vec3ui::Z, vec3ui::X, vec3ui::Y, AO1, AO2, AO3, AO4);
            shad_col = col_vec3r * shadowZneg;
            m_colorArray->push_back(shad_col * AO1);
            m_colorArray->push_back(shad_col * AO2);
            m_colorArray->push_back(shad_col * AO3);
            m_colorArray->push_back(shad_col * AO4);
            osg::Vec3 normal(0.0f, 0.0f, -1.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
        }
      }

  setVertexArray(m_vertexArray);
  setColorArray(m_colorArray);
  setNormalArray(m_normalsArray);
  setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

  osg::DrawArrays *drawArrays = static_cast<osg::DrawArrays *>(getPrimitiveSet(0));
  drawArrays->setCount(m_vertexArray->getNumElements());
  drawArrays->dirty();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 10.0f);
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 10.0f);
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.2f);
  mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);

  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);
  // stateset->setMode(GL_COLOR_MATERIAL, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}
