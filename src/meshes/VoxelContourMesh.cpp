#include "VoxelContourMesh.hpp"

VoxelContourMesh::VoxelContourMesh(VoxelArray* voxels) : VoxelMesh(voxels) {
  m_texture = new osg::Texture2D;
  m_image = osgDB::readImageFile("assets/voxel.png");
  m_texture->setImage(m_image);

  m_texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
  m_texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
  m_texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
  // m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
  //                      osg::Texture2D::FilterMode::NEAREST);
  // m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
  //                      osg::Texture2D::FilterMode::NEAREST);
  m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
                       osg::Texture2D::FilterMode::LINEAR);
  m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
                       osg::Texture2D::FilterMode::LINEAR);
}

VoxelContourMesh::VoxelContourMesh(const VoxelMesh& voxMesh)
    : VoxelMesh(voxMesh) {
  m_texture = new osg::Texture2D;
  m_image = osgDB::readImageFile("assets/voxel.png");
  m_texture->setImage(m_image);

  m_texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
  m_texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
  m_texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
  // m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
  //                      osg::Texture2D::FilterMode::NEAREST);
  // m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
  //                      osg::Texture2D::FilterMode::NEAREST);
  m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
                       osg::Texture2D::FilterMode::LINEAR);
  m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
                       osg::Texture2D::FilterMode::LINEAR);
}

// build the mesh for the voxel array
void VoxelContourMesh::build() {
  for (int i = 0; i < m_arrayOrig->m_colors->getNumElements(); i++)
    m_arrayOrig->m_colors->at(i) = osg::Vec4(1, 1, 1, 1);
  for (int i = 0; i < m_arrayTmp1->m_colors->getNumElements(); i++)
    m_arrayTmp1->m_colors->at(i) = osg::Vec4(1, 1, 1, 1);
  for (int i = 0; i < m_arrayTmp2->m_colors->getNumElements(); i++)
    m_arrayTmp2->m_colors->at(i) = osg::Vec4(1, 1, 1, 1);

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();

  // Transparent alpha channel
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  stateset->setTextureAttribute(0, m_texture, osg::StateAttribute::OVERRIDE);
  stateset->setTextureMode(
      0, GL_TEXTURE_2D,
      osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setColorMode(osg::Material::ColorMode::EMISSION);

  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

  osg::Vec3i voxMin(2, 2, 2);
  osg::Vec3i voxMax(getSizeX() - 3, getSizeY() - 3, getSizeZ() - 3);
  crop(voxMin, voxMax);
}
