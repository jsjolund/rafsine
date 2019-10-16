#include "VoxelAreaMesh.hpp"

MeshArray VoxelAreaMesh::createBox(osg::Vec3i min,
                                   osg::Vec3i max,
                                   osg::Vec4 color) {
  MeshArray meshArray;
  osg::Vec3i s = max - min;
  osg::Vec2Array* texCoords = meshArray.m_texCoords;
  texCoords->push_back(osg::Vec2(0, 0));
  texCoords->push_back(osg::Vec2(s.x(), 0));
  texCoords->push_back(osg::Vec2(s.x(), s.z()));
  texCoords->push_back(osg::Vec2(0, s.z()));

  texCoords->push_back(osg::Vec2(0, 0));
  texCoords->push_back(osg::Vec2(s.x(), 0));
  texCoords->push_back(osg::Vec2(s.x(), s.y()));
  texCoords->push_back(osg::Vec2(0, s.y()));

  texCoords->push_back(osg::Vec2(0, 0));
  texCoords->push_back(osg::Vec2(s.y(), 0));
  texCoords->push_back(osg::Vec2(s.y(), s.z()));
  texCoords->push_back(osg::Vec2(0, s.z()));

  texCoords->push_back(osg::Vec2(0, 0));
  texCoords->push_back(osg::Vec2(s.y(), 0));
  texCoords->push_back(osg::Vec2(s.y(), s.z()));
  texCoords->push_back(osg::Vec2(0, s.z()));

  texCoords->push_back(osg::Vec2(0, 0));
  texCoords->push_back(osg::Vec2(s.x(), 0));
  texCoords->push_back(osg::Vec2(s.x(), s.z()));
  texCoords->push_back(osg::Vec2(0, s.z()));

  texCoords->push_back(osg::Vec2(0, 0));
  texCoords->push_back(osg::Vec2(s.x(), 0));
  texCoords->push_back(osg::Vec2(s.x(), s.y()));
  texCoords->push_back(osg::Vec2(0, s.y()));

  osg::Vec3Array* normals = meshArray.m_normals;
  normals->push_back(osg::Vec3(0, -1, 0));
  normals->push_back(osg::Vec3(0, -1, 0));
  normals->push_back(osg::Vec3(0, -1, 0));
  normals->push_back(osg::Vec3(0, -1, 0));

  normals->push_back(osg::Vec3(0, 0, 1));
  normals->push_back(osg::Vec3(0, 0, 1));
  normals->push_back(osg::Vec3(0, 0, 1));
  normals->push_back(osg::Vec3(0, 0, 1));

  normals->push_back(osg::Vec3(-1, 0, 0));
  normals->push_back(osg::Vec3(-1, 0, 0));
  normals->push_back(osg::Vec3(-1, 0, 0));
  normals->push_back(osg::Vec3(-1, 0, 0));

  normals->push_back(osg::Vec3(1, 0, 0));
  normals->push_back(osg::Vec3(1, 0, 0));
  normals->push_back(osg::Vec3(1, 0, 0));
  normals->push_back(osg::Vec3(1, 0, 0));

  normals->push_back(osg::Vec3(0, 1, 0));
  normals->push_back(osg::Vec3(0, 1, 0));
  normals->push_back(osg::Vec3(0, 1, 0));
  normals->push_back(osg::Vec3(0, 1, 0));

  normals->push_back(osg::Vec3(0, 0, -1));
  normals->push_back(osg::Vec3(0, 0, -1));
  normals->push_back(osg::Vec3(0, 0, -1));
  normals->push_back(osg::Vec3(0, 0, -1));

  osg::Vec3Array* vertices = meshArray.m_vertices;
  vertices->push_back(osg::Vec3(min.x(), min.y(), min.z()));  // 0
  vertices->push_back(osg::Vec3(max.x(), min.y(), min.z()));  // 3
  vertices->push_back(osg::Vec3(max.x(), min.y(), max.z()));  // 5
  vertices->push_back(osg::Vec3(min.x(), min.y(), max.z()));  // 1

  vertices->push_back(osg::Vec3(min.x(), min.y(), max.z()));  // 1
  vertices->push_back(osg::Vec3(max.x(), min.y(), max.z()));  // 5
  vertices->push_back(osg::Vec3(max.x(), max.y(), max.z()));  // 7
  vertices->push_back(osg::Vec3(min.x(), max.y(), max.z()));  // 4

  vertices->push_back(osg::Vec3(min.x(), max.y(), min.z()));  // 2
  vertices->push_back(osg::Vec3(min.x(), min.y(), min.z()));  // 0
  vertices->push_back(osg::Vec3(min.x(), min.y(), max.z()));  // 1
  vertices->push_back(osg::Vec3(min.x(), max.y(), max.z()));  // 4

  vertices->push_back(osg::Vec3(max.x(), min.y(), min.z()));  // 3
  vertices->push_back(osg::Vec3(max.x(), max.y(), min.z()));  // 6
  vertices->push_back(osg::Vec3(max.x(), max.y(), max.z()));  // 7
  vertices->push_back(osg::Vec3(max.x(), min.y(), max.z()));  // 5

  vertices->push_back(osg::Vec3(max.x(), max.y(), min.z()));  // 6
  vertices->push_back(osg::Vec3(min.x(), max.y(), min.z()));  // 2
  vertices->push_back(osg::Vec3(min.x(), max.y(), max.z()));  // 4
  vertices->push_back(osg::Vec3(max.x(), max.y(), max.z()));  // 7

  vertices->push_back(osg::Vec3(max.x(), min.y(), min.z()));  // 3
  vertices->push_back(osg::Vec3(min.x(), min.y(), min.z()));  // 0
  vertices->push_back(osg::Vec3(min.x(), max.y(), min.z()));  // 2
  vertices->push_back(osg::Vec3(max.x(), max.y(), min.z()));  // 6

  osg::Vec4Array* colors = meshArray.m_colors;
  for (int i = 0; i < 24; i++) colors->push_back(color);

  return meshArray;
}

VoxelAreaMesh::VoxelAreaMesh(osg::Vec3i min, osg::Vec3i max) : osg::Geometry() {
  setUseVertexBufferObjects(true);
  addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0));

  m_texture = new osg::Texture2D;
  m_image = osgDB::readImageFile("assets/voxel.png");
  // m_image = osgDB::readImageFile("assets/logo.png");
  m_texture->setImage(m_image);
  m_texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
  m_texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
  m_texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
  m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
                       osg::Texture2D::FilterMode::LINEAR);
  m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
                       osg::Texture2D::FilterMode::LINEAR);

  m_array = createBox(min, max, osg::Vec4f(1, 1, 1, 0.5));
  m_array.dirty();

  setVertexArray(m_array.m_vertices);
  setNormalArray(m_array.m_normals, osg::Array::BIND_PER_VERTEX);
  setColorArray(m_array.m_colors, osg::Array::BIND_PER_VERTEX);
  setTexCoordArray(0, m_array.m_texCoords, osg::Array::BIND_PER_VERTEX);

  osg::DrawArrays* drawArrays =
      static_cast<osg::DrawArrays*>(getPrimitiveSet(0));
  drawArrays->setCount(m_array.m_vertices->getNumElements());
  drawArrays->dirty();

  osg::ref_ptr<osg::StateSet> stateset = getOrCreateStateSet();

  // Transparent alpha channel
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
  stateset->setTextureAttribute(0, m_texture, osg::StateAttribute::OVERRIDE);
  stateset->setTextureMode(
      0, GL_TEXTURE_2D,
      osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
  mat->setColorMode(osg::Material::ColorMode::EMISSION);
  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);
}
