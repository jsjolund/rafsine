#include "StlMesh.hpp"

StlMesh::StlMesh(const stl_file::StlFile& solid, const osg::Vec4 color) {
  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();

  // Set triangle vertices
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  for (int i = 0; i < solid.vertices.size(); i += 3) {
    float v0x = solid.vertices.at(i + 0);
    float v0y = solid.vertices.at(i + 1);
    float v0z = solid.vertices.at(i + 2);
    vertices->push_back(osg::Vec3(v0x, v0y, v0z));
  }
  geometry->addPrimitiveSet(
      new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, vertices->size()));
  geometry->setVertexArray(vertices);

  // Set triangle normals
  osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
  for (int i = 0; i < solid.normals.size(); i += 3) {
    float n0 = solid.normals.at(i);
    float n1 = solid.normals.at(i + 1);
    float n2 = solid.normals.at(i + 2);
    normals->push_back(osg::Vec3(n0, n1, n2));
    normals->push_back(osg::Vec3(n0, n1, n2));
    normals->push_back(osg::Vec3(n0, n1, n2));
  }
  geometry->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);

  // Set color
  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
  colors->push_back(color);
  geometry->setColorArray(colors, osg::Array::BIND_OVERALL);

  // Enable depth and lighting mode
  osg::ref_ptr<osg::StateSet> stateset = geometry->getOrCreateStateSet();
  stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);

  // Set material
  osg::ref_ptr<osg::Material> mat = new osg::Material();
  mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
  mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                  osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.5f);
  mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                   osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
  mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
  stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

  addDrawable(geometry);
}