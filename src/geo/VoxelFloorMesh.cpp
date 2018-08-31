#include "VoxelFloorMesh.hpp"

VoxelFloorMesh::VoxelFloorMesh(VoxelArray *voxels)
    : m_width(voxels->getSizeX()),
      m_height(voxels->getSizeY()),
      m_voxels(voxels),
      m_transform(new osg::PositionAttitudeTransform()),
      osg::Geometry(*osg::createTexturedQuadGeometry(
                        osg::Vec3(0.0f, 0.0f, 0.0f),
                        osg::Vec3(voxels->getSizeX(), 0.0f, 0.0f),
                        osg::Vec3(0.0f, 0.0f, voxels->getSizeY()),
                        0.0f,
                        0.0f,
                        1.0f,
                        1.0f),
                    osg::CopyOp::SHALLOW_COPY)
{
  m_texture = new osg::Texture2D();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
  stateset->setTextureAttribute(0, m_texture, osg::StateAttribute::ON);
  stateset->setTextureMode(0, GL_TEXTURE_2D, osg::StateAttribute::ON);

  m_image = new osg::Image();
  m_image->allocateImage(m_width, m_height, 1, GL_RGB, GL_UNSIGNED_BYTE, 1);

  m_texture->setDataVariance(osg::Object::DYNAMIC);
  m_texture->setResizeNonPowerOfTwoHint(true);
  m_texture->setBorderWidth(0);
  m_texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
  m_texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
  m_texture->setTextureSize(m_width, m_height);
  m_texture->setSourceFormat(GL_RGB);
  m_texture->setSourceType(GL_UNSIGNED_BYTE);
  m_texture->setInternalFormat(GL_RGB8);
  m_texture->setImage(m_image);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  geode->addDrawable(this);
  m_transform->addChild(geode);

  GLubyte *imgDataPtr = m_image->data();
  int format = m_image->getPixelSizeInBits();
  //background color
  col3 bc = col3::black;
  //line color
  col3 c0 = col3(253, 254, 210);
  col3 c1 = col3(36, 49, 52);
  col3 c2 = col3(48, 91, 116);
  col3 c3 = col3(104, 197, 214);
  //conversion factors
  double Cx = m_voxels->getSizeX() / double(m_texture->getTextureWidth());
  double Cy = m_voxels->getSizeY() / double(m_texture->getTextureHeight());
  for (unsigned int i = 0; i < m_texture->getTextureWidth(); i++)
    for (unsigned int j = 0; j < m_texture->getTextureHeight(); j++)
    {
      int offset = (j * m_width + i) * format / 8;
      int x = i * Cx;
      int y = j * Cy;
      *(imgDataPtr + offset + 0) = bc.r;
      *(imgDataPtr + offset + 1) = bc.g;
      *(imgDataPtr + offset + 2) = bc.b;
      if (i % 5 == 0)
      {
        *(imgDataPtr + offset + 0) = c1.r;
        *(imgDataPtr + offset + 1) = c1.g;
        *(imgDataPtr + offset + 2) = c1.b;
      }
      if (j % 5 == 0)
      {
        *(imgDataPtr + offset + 0) = c1.r;
        *(imgDataPtr + offset + 1) = c1.g;
        *(imgDataPtr + offset + 2) = c1.b;
      }
      if (((i % 20 < 2) || (i % 20 > 18)) && ((j % 20 < 2) || (j % 20 > 18)))
      {
        *(imgDataPtr + offset + 0) = c2.r;
        *(imgDataPtr + offset + 1) = c2.g;
        *(imgDataPtr + offset + 2) = c2.b;
      }
      if ((m_voxels->isEmptyStrict(x, y, 0)) && (!(m_voxels->isEmptyStrict(x + 1, y, 0)) ||
                                                 !(m_voxels->isEmptyStrict(x - 1, y, 0)) ||
                                                 !(m_voxels->isEmptyStrict(x, y + 1, 0)) ||
                                                 !(m_voxels->isEmptyStrict(x, y - 1, 0))))
      {
        *(imgDataPtr + offset + 0) = c0.r;
        *(imgDataPtr + offset + 1) = c0.g;
        *(imgDataPtr + offset + 2) = c0.b;
      }
    }
  m_image->dirty();
}
