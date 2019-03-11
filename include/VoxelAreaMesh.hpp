#pragma once

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/TexMat>
#include <osg/Texture2D>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osgDB/ReadFile>

#include <glm/vec3.hpp>

#include "ColorSet.hpp"

class VoxelAreaMesh : public osg::Geode {
 private:
  osg::ref_ptr<osg::Texture2D> m_texture;
  osg::ref_ptr<osg::Image> m_image;
  ColorSet m_colorSet;

 public:
  VoxelAreaMesh(glm::ivec3 min, glm::ivec3 max) : osg::Geode() {
    m_texture = new osg::Texture2D;
    m_image = osgDB::readImageFile("assets/voxel.png");
    m_texture->setImage(m_image);
    m_texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
    m_texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    m_texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
    m_texture->setFilter(osg::Texture2D::FilterParameter::MIN_FILTER,
                         osg::Texture2D::FilterMode::LINEAR);
    m_texture->setFilter(osg::Texture2D::FilterParameter::MAG_FILTER,
                         osg::Texture2D::FilterMode::LINEAR);

    glm::ivec3 size = max - min;
    glm::vec3 c =
        glm::vec3(min) + glm::vec3(size.x * 0.5f, size.y * 0.5f, size.z * 0.5f);

    osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(
        new osg::Box(osg::Vec3d(c.x, c.y, c.z), size.x, size.y, size.z));
    osg::Vec4 color = m_colorSet.getColor(10);
    // color.a() *= 0.5;
    drawable->setColor(color);
    addDrawable(drawable);

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
                    osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
    mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                    osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
    mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                     osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
    mat->setColorMode(osg::Material::ColorMode::EMISSION);
    stateset->setAttribute(mat.get(), osg::StateAttribute::Values::ON);

    osg::ref_ptr<osg::TexMat> tm = new osg::TexMat;
    tm->setMatrix(osg::Matrix::scale(size.x, size.y, size.z));
    stateset->setAttributeAndModes(tm, osg::StateAttribute::ON);
  }
};
