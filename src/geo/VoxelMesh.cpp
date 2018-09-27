#include "VoxelMesh.hpp"

// Constructor from a file on the disk
// TODO: to be modified with the ressource manager
VoxelMesh::VoxelMesh(std::string voxel_file_name)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform())
{
  //load file size
  std::ifstream fin(voxel_file_name.c_str());
  unsigned int nx, ny, nz;
  fin >> nx >> ny >> nz;
  fin.close();

  // TODO: This should be deleted in destructor?
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
VoxelMesh::VoxelMesh(VoxelArray *voxels)
    : osg::Geometry(),
      m_transform(new osg::PositionAttitudeTransform()),
      m_voxels(voxels)
{
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
      m_normalsArray(voxmesh.m_normalsArray)
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
  return *this;
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

  std::cout << "min: "
            << voxMin.x() << ", " << voxMin.y() << ", " << voxMin.z() << " max: "
            << voxMax.x() << ", " << voxMax.y() << ", " << voxMax.z() << ", vox: "
            << m_voxels->getSizeX() << ", " << m_voxels->getSizeY() << ", " << m_voxels->getSizeZ()
            << std::endl;

  for (int k = 0; k < int(m_voxels->getSizeZ()); ++k)
  {
    for (int j = 0; j < int(m_voxels->getSizeY()); ++j)
    {
      for (int i = 0; i < int(m_voxels->getSizeX()); ++i)
      {
        if (i < voxMin.x() || (j < voxMin.y()) || (k < voxMin.z()) || (i > voxMax.x()) || (j > voxMax.y()) || (k > voxMax.z()))
          continue;
        if (!m_voxels->isEmpty(i, j, k))
        {
          voxel v = m_voxels->getVoxelReadOnly(i, j, k);
          osg::Vec4 color = m_colorSet->getColor(v);

          if (m_voxels->isEmpty(i + 1, j, k))
          {
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k));
            m_vertexArray->push_back(osg::Vec3(i + 1, j + 1, k + 1));
            m_vertexArray->push_back(osg::Vec3(i + 1, j, k + 1));
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
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
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
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
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
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
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
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
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
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
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            m_colorArray->push_back(color);
            osg::Vec3 normal(0.0f, 0.0f, -1.0f);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
            m_normalsArray->push_back(normal);
          }
        }
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
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON | osg::StateAttribute::PROTECTED);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 2.0f);
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.5f);
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
  mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);

  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);
  // stateset->setMode(GL_COLOR_MATERIAL, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);

  if (m_transform->getNumChildren() > 0)
    m_transform->removeChildren(0, m_transform->getNumChildren());
  m_transform->addChild(geode);
}
