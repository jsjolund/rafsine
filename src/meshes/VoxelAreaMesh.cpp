#include "VoxelAreaMesh.hpp"

MeshArray VoxelAreaMesh::createBox(glm::ivec3 min, glm::ivec3 max,
                                   glm::ivec4 color) {
  MeshArray meshArray;

  // osg::Vec2Array* texCoords = meshArray.m_texCoords;
  // texCoords->push_back(osg::Vec2(0, 0));
  // texCoords->push_back(osg::Vec2(max.x-min.x, 0));
  // texCoords->push_back(osg::Vec2(max.x-min.x, 0));

  osg::Vec3Array* vertices = meshArray.m_vertices;
  vertices->push_back(osg::Vec3(min.x, min.y, min.z));  // 0
  vertices->push_back(osg::Vec3(max.x, min.y, min.z));  // 3
  vertices->push_back(osg::Vec3(max.x, min.y, max.z));  // 5
  vertices->push_back(osg::Vec3(min.x, min.y, max.z));  // 1

  vertices->push_back(osg::Vec3(min.x, min.y, max.z));  // 1
  vertices->push_back(osg::Vec3(max.x, min.y, max.z));  // 5
  vertices->push_back(osg::Vec3(max.x, max.y, max.z));  // 7
  vertices->push_back(osg::Vec3(min.x, max.y, max.z));  // 4

  vertices->push_back(osg::Vec3(min.x, min.y, min.z));  // 0
  vertices->push_back(osg::Vec3(min.x, min.y, max.z));  // 1
  vertices->push_back(osg::Vec3(min.x, max.y, max.z));  // 4
  vertices->push_back(osg::Vec3(min.x, max.y, min.z));  // 2

  vertices->push_back(osg::Vec3(max.x, min.y, min.z));  // 3
  vertices->push_back(osg::Vec3(max.x, max.y, min.z));  // 6
  vertices->push_back(osg::Vec3(max.x, max.y, max.z));  // 7
  vertices->push_back(osg::Vec3(max.x, min.y, max.z));  // 5

  vertices->push_back(osg::Vec3(max.x, max.y, min.z));  // 6
  vertices->push_back(osg::Vec3(min.x, max.y, min.z));  // 2
  vertices->push_back(osg::Vec3(min.x, max.y, max.z));  // 4
  vertices->push_back(osg::Vec3(max.x, max.y, max.z));  // 7

  vertices->push_back(osg::Vec3(max.x, min.y, min.z));  // 3
  vertices->push_back(osg::Vec3(min.x, min.y, min.z));  // 0
  vertices->push_back(osg::Vec3(min.x, max.y, min.z));  // 2
  vertices->push_back(osg::Vec3(max.x, max.y, min.z));  // 6

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
}