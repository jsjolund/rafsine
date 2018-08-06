#include "CFDScene.hpp"

//increase slice x position
void CFDScene::sliceXup()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.x < (long)vox_size_.x - 1)
      slice_pos_.x++;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.x < (long)vox_size_.x)
      vox_min_.x++;
    redrawVoxelMesh();
    break;
  }
}

//decrease slice x position
void CFDScene::sliceXdown()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.x > 1)
      slice_pos_.x--;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.x >= 0)
      vox_min_.x--;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::sliceYup()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.y < (long)vox_size_.y - 1)
      slice_pos_.y++;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.y < (long)vox_size_.y)
      vox_min_.y++;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::sliceYdown()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.y > 1)
      slice_pos_.y--;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_min_.y >= 0)
      vox_min_.y--;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::sliceZup()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.z < (long)vox_size_.z - 1)
      slice_pos_.z++;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_max_.z < (long)vox_size_.z)
      vox_max_.z++;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::sliceZdown()
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if (slice_pos_.z > 1)
      slice_pos_.z--;
    break;
  case DisplayMode::VOX_GEOMETRY:
    if (vox_max_.z >= 0)
      vox_max_.z--;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::setSliceXpos(int pos)
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if ((pos >= 0) && (pos < (long)vox_size_.x))
      slice_pos_.x = pos;
    break;
  case DisplayMode::VOX_GEOMETRY:
    vox_min_.x = pos;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::setSliceYpos(int pos)
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if ((pos >= 0) && (pos < (long)vox_size_.y))
      slice_pos_.y = pos;
    break;
  case DisplayMode::VOX_GEOMETRY:
    vox_min_.y = pos;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::setSliceZpos(int pos)
{
  switch (displayMode_)
  {
  case DisplayMode::SLICE:
    if ((pos >= 0) && (pos < (long)vox_size_.z))
      slice_pos_.z = pos;
    break;
  case DisplayMode::VOX_GEOMETRY:
    vox_max_.z = pos;
    redrawVoxelMesh();
    break;
  }
}

void CFDScene::redrawVoxelMesh()
{
  voxmesh_->buildMesh(vox_min_.x, vox_max_.x,
                      vox_min_.y, vox_max_.y,
                      vox_min_.z, vox_max_.z);
  voxGeo->setVertexArray(voxmesh_->vertices_);
  voxGeo->setColorArray(voxmesh_->v_colors_);
  voxGeo->setNormalArray(voxmesh_->normals_);
  voxGeo->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
  voxGeo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  if (voxGeo->getPrimitiveSetList().size() > 0)
    voxGeo->removePrimitiveSet(0);
  voxGeo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, voxmesh_->vertices_->getNumElements()));
std::cout << "redraw vox" << std::endl;
}

void CFDScene::setVoxelMesh(VoxelMesh *mesh)
{
  voxmesh_ = mesh;
  vox_size_ = vec3<int>(voxmesh_->getSizeX(), voxmesh_->getSizeY(), voxmesh_->getSizeZ());
  slice_pos_ = vox_size_ / 2;
  vox_min_ = vec3<int>(-1, -1, -1);
  vox_max_ = vox_size_;
  std::cout << "set vox" << std::endl;
  redrawVoxelMesh();
}

osg::Vec3 CFDScene::getCenter()
{
  return osg::Vec3(vox_size_.x / 2, vox_size_.y / 2, vox_size_.z / 2);
}

CFDScene::CFDScene()
{
  root_ = new osg::Group();
  osg::Geode *geode = new osg::Geode();
  voxGeo = new osg::Geometry();
  voxGeoTransform = new osg::PositionAttitudeTransform();

  geode->addDrawable(voxGeo);
  voxGeoTransform->addChild(geode);
  root_->addChild(voxGeoTransform);

  voxGeoTransform->setPosition(osg::Vec3(0, 0, 0));

  osg::Group *lightGroup = new osg::Group();
  osg::LightSource *lightSource = new osg::LightSource;
  osg::Light *light = new osg::Light;
  light->setAmbient(osg::Vec4(1.0, 1.0, 1.0, 1.0));
  light->setDiffuse(osg::Vec4(1.0, 1.0, 1.0, 1.0));
  light->setSpecular(osg::Vec4(1, 1, 1, 1));

  osg::Quat q = osg::Quat(osg::PI / 4, osg::Vec3d(1, 0, 0),
                          0, osg::Vec3d(0, 1, 0),
                          osg::PI / 4, osg::Vec3d(0, 0, 1));

  light->setDirection(q * osg::Vec3(1.0f, 0.0f, 0.0f));
  lightSource->setLight(light);
  lightGroup->addChild(lightSource);
  root_->addChild(lightGroup);
  std::cout << "create scene" << std::endl;
}
