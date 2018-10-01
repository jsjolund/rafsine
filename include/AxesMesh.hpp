#pragma once

#include <osg/Material>
#include <osg/Node>
#include <osg/StateSet>
#include <osg/PositionAttitudeTransform>
#include <osgDB/ReadFile>

class AxesMesh : public osg::PositionAttitudeTransform
{
public:
  inline AxesMesh()
      : osg::PositionAttitudeTransform()
  {
    osg::ref_ptr<osg::Node> axesModel = osgDB::readNodeFile("assets/axes.osgt");

    osg::ref_ptr<osg::StateSet> stateset = axesModel->getOrCreateStateSet();
    // stateset->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setAmbient(osg::Material::Face::FRONT_AND_BACK,
                    osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
    mat->setDiffuse(osg::Material::Face::FRONT_AND_BACK,
                    osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 1.0f);
    mat->setEmission(osg::Material::Face::FRONT_AND_BACK,
                     osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) * 0.1f);
    mat->setColorMode(osg::Material::ColorMode::AMBIENT_AND_DIFFUSE);
    stateset->setAttributeAndModes(mat, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    addChild(axesModel);
    setScale(osg::Vec3d(100, 100, 100));
  }
};