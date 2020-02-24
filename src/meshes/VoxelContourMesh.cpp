#include "VoxelContourMesh.hpp"

VoxelContourMesh::VoxelContourMesh(const VoxelMesh& voxMesh)
    : VoxelMesh(voxMesh) {
  // m_texture = new osg::Texture2D;
  // m_image = osgDB::readImageFile("assets/voxel.png");
  // m_texture->setImage(m_image);
  // m_texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
  // m_texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
  // m_texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
  // m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
  //                      osg::Texture2D::FilterMode::LINEAR);
  // m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
  //                      osg::Texture2D::FilterMode::LINEAR);
  build();
}

void VoxelContourMesh::build() {
  osg::ref_ptr<osg::StateSet> stateset = m_geo->getOrCreateStateSet();

  osg::Vec3i voxMin(2, 2, 2);
  osg::Vec3i voxMax(getSizeX() - 3, getSizeY() - 3, getSizeZ() - 3);
  crop(voxMin, voxMax);

  // Transparent alpha channel
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  stateset->setAttributeAndModes(
      new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

  // Set to uniformly colored lines
  osg::ref_ptr<osg::Material> material = new osg::Material;
  material->setColorMode(osg::Material::OFF);

  float alpha = 1.0f;
  float shade = 0.75f;
  material->setAmbient(osg::Material::FRONT_AND_BACK,
                       osg::Vec4(0.0f, 0.0f, 0.0f, alpha));
  material->setDiffuse(osg::Material::FRONT_AND_BACK,
                       osg::Vec4(0.0f, 0.0f, 0.0f, alpha));
  material->setSpecular(osg::Material::FRONT_AND_BACK,
                        osg::Vec4(0.0f, 0.0f, 0.0f, alpha));
  material->setEmission(osg::Material::FRONT_AND_BACK,
                        osg::Vec4(shade, shade, shade, alpha));

  stateset->setAttributeAndModes(
      material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING,
                    osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

  osg::ref_ptr<osg::PolygonOffset> polyoffset = new osg::PolygonOffset;
  polyoffset->setFactor(-1.0f);
  polyoffset->setUnits(-1.0f);
  osg::ref_ptr<osg::PolygonMode> polymode = new osg::PolygonMode;
  polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
  stateset->setAttributeAndModes(
      polyoffset, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
  stateset->setAttributeAndModes(
      polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

  setPolygonMode(osg::PolygonMode::LINE);
}
