#pragma once

#include <osg/Array>
#include <osg/CopyOp>

#include <utility>

class MeshArray {
 public:
  //! Vertices of mesh
  osg::ref_ptr<osg::Vec3Array> m_vertices;
  //! Color of each vertex
  osg::ref_ptr<osg::Vec4Array> m_colors;
  //! Plane normals
  osg::ref_ptr<osg::Vec3Array> m_normals;
  //! Texture coordinates
  osg::ref_ptr<osg::Vec2Array> m_texCoords;

  MeshArray(const MeshArray& other)
      : m_vertices(
            new osg::Vec3Array(*other.m_vertices, osg::CopyOp::DEEP_COPY_ALL)),
        m_colors(
            new osg::Vec4Array(*other.m_colors, osg::CopyOp::DEEP_COPY_ALL)),
        m_normals(
            new osg::Vec3Array(*other.m_normals, osg::CopyOp::DEEP_COPY_ALL)),
        m_texCoords(new osg::Vec2Array(*other.m_texCoords,
                                       osg::CopyOp::DEEP_COPY_ALL)) {
    dirty();
  }

  MeshArray()
      : m_vertices(new osg::Vec3Array()),
        m_colors(new osg::Vec4Array()),
        m_normals(new osg::Vec3Array()),
        m_texCoords(new osg::Vec2Array()) {}

  size_t size() { return m_vertices->size(); }

  void dirty() {
    m_vertices->dirty();
    m_colors->dirty();
    m_normals->dirty();
    m_texCoords->dirty();
  }

  void erase(int first, int last) {
    m_vertices->erase(m_vertices->begin() + first, m_vertices->begin() + last);
    m_colors->erase(m_colors->begin() + first, m_colors->begin() + last);
    m_normals->erase(m_normals->begin() + first, m_normals->begin() + last);
    m_texCoords->erase(m_texCoords->begin() + first,
                       m_texCoords->begin() + last);
  }

  void insert(MeshArray* other) {
    m_vertices->insert(m_vertices->end(), other->m_vertices->begin(),
                       other->m_vertices->end());
    m_colors->insert(m_colors->end(), other->m_colors->begin(),
                     other->m_colors->end());
    m_normals->insert(m_normals->end(), other->m_normals->begin(),
                      other->m_normals->end());
    m_texCoords->insert(m_texCoords->end(), other->m_texCoords->begin(),
                        other->m_texCoords->end());
  }

  void clear() {
    m_vertices->clear();
    m_colors->clear();
    m_normals->clear();
    m_texCoords->clear();

    m_vertices->trim();
    m_colors->trim();
    m_normals->trim();
    m_texCoords->trim();
  }

  static void swap(MeshArray* f1, MeshArray* f2) {
    std::swap(f1->m_vertices, f2->m_vertices);
    std::swap(f1->m_colors, f2->m_colors);
    std::swap(f1->m_normals, f2->m_normals);
    std::swap(f1->m_texCoords, f2->m_texCoords);
  }
};
