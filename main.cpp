#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgDB/ReadFile>

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>

#include <iostream>

#include "gui/ResizeBar.hpp"
#include "gui/GLWindow.hpp"
#include "gui/TreeView.hpp"
#include "gui/DataTable.hpp"
#include "gui/MainMenu.hpp"
#include "geo/VoxelGeometry.hpp"
#include "geo/Voxel.hpp"
#include "geo/VoxelMesh.hpp"

#include "ext/st_tree/include/st_tree.h"

// APP WINDOW CLASS
class MyAppWindow : public Fl_Double_Window
{
  public:
    GLWindow *mygl;

  private:
    const int resize_h = 5;
    const int resize_w = 5;
    const int menu_h = 24;
    TreeView *tree;
    DataTable *table;
    ResizerBarHoriz *hbar;
    ResizerBarVert *vbar;
    MainMenu *menu;

  public:
    void resize(int X, int Y, int W, int H)
    {
        Fl_Double_Window::resize(X, Y, W, H);

        menu->resize(0, 0, W, menu_h);
        tree->resize(0,
                     menu_h,
                     tree->w(),
                     tree->h());
        hbar->resize(0,
                     tree->y() + tree->h(),
                     tree->w(),
                     resize_h);
        table->resize(0,
                      hbar->y() + hbar->h(),
                      hbar->w(),
                      h() - (hbar->y() + hbar->h()));
        vbar->resize(tree->x() + tree->w(),
                     menu_h,
                     resize_w,
                     h());
        mygl->resize(vbar->x() + vbar->w(),
                     menu_h,
                     w() - tree->w(),
                     h());
    }

    MyAppWindow(int W, int H, const char *L = 0) : Fl_Double_Window(W, H, L)
    {
        menu = new MainMenu(0, 0, W, menu_h);
        tree = new TreeView(0,
                            menu_h,
                            int(float(w()) * 1 / 3 - resize_w),
                            int(float(h()) * 1 / 2 - resize_h));
        hbar = new ResizerBarHoriz(0,
                                   tree->y() + tree->h(),
                                   tree->w(),
                                   resize_h);
        table = new DataTable(0,
                              hbar->y() + hbar->h(),
                              hbar->w(),
                              h() - (hbar->y() + hbar->h()));
        vbar = new ResizerBarVert(tree->x() + tree->w(),
                                  menu_h,
                                  resize_w,
                                  h());
        mygl = new GLWindow(vbar->x() + vbar->w(),
                            menu_h,
                            w() - tree->w(),
                            h());
        end();
    }
};

void idle_cb()
{
    Fl::redraw();
}

// osg::Node *createScene()
// {
//     // create the Geode (Geometry Node) to contain all our osg::Geometry objects.
//     osg::Geode *geode = new osg::Geode();

//     osg::ref_ptr<osg::Vec4Array> shared_colors = new osg::Vec4Array;
//     shared_colors->push_back(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f));

//     // Same trick for shared normal.
//     osg::ref_ptr<osg::Vec3Array> shared_normals = new osg::Vec3Array;
//     shared_normals->push_back(osg::Vec3(0.0f, -1.0f, 0.0f));

//     // create QUADS
//     {
//         // create Geometry object to store all the vertices and lines primitive.
//         osg::Geometry *polyGeom = new osg::Geometry();

//         int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);

//         osg::Vec3Array *vertices = new osg::Vec3Array(numCoords, myCoords);

//         // pass the created vertex array to the points geometry object.
//         polyGeom->setVertexArray(vertices);

//         // use the shared color array.
//         polyGeom->setColorArray(shared_colors.get(), osg::Array::BIND_OVERALL);

//         // use the shared normal array.
//         polyGeom->setNormalArray(shared_normals.get(), osg::Array::BIND_OVERALL);

//         // This time we simply use primitive, and hardwire the number of coords to use
//         // since we know up front,
//         polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, numCoords));

//         // add the points geometry to the geode.
//         geode->addDrawable(polyGeom);
//     }
//     return geode;
// }

int main(int argc, char **argv)
{
    UnitConverter uc(6.95, 15, 1.0, 0.1, 1, 0, 0);
    real mx = 6.95;
    real my = 6.4;
    real mz = 3.1;
    VoxelGeometry vox(uc.m_to_lu(mx) + 1, uc.m_to_lu(my) + 1, uc.m_to_lu(mz) + 1, &uc);
    vox.addWallXmin();
    vox.addWallYmin();
    vox.addWallZmin();
    vox.addWallXmax();
    vox.addWallYmax();
    vox.addWallZmax();

    DomainGeometryBox box("test", vec3<real>(1, 1, 0), vec3<real>(3, 3, 2));
    vox.addSolidBox(&box);
    vox.saveToFile("test.vox");
    VoxelGeometry vox1;

    vox1.loadFromFile("test.vox");
    std::cout << vox1 << std::endl;

    VoxelMesh mesh(*(vox.data));
    mesh.buildMesh();

    osg::Group *root = new osg::Group();
    osg::Geode *geode = new osg::Geode();
    osg::Geometry *polyGeom = new osg::Geometry();
    polyGeom->setVertexArray(mesh.vertices_);
    polyGeom->setColorArray(mesh.v_colors_, osg::Array::BIND_OVERALL);
    int numQuads = mesh.vertices_->getNumElements()/4;
    polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, numQuads));
    geode->addDrawable(polyGeom);
    // geode->getOrCreateStateSet()->setMode(GL_LIGHTING,
    //   osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
    // geode->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    // geode->getOrCreateStateSet()->setMode(GL_TEXTURE_2D, osg::StateAttribute::OFF);
    root->addChild(geode);

    MyAppWindow appWindow(1280, 720, "My app");
    appWindow.resizable(&appWindow);

    appWindow.mygl->setSceneData(root);
    appWindow.mygl->setCameraManipulator(new osgGA::TrackballManipulator);
    appWindow.mygl->addEventHandler(new osgViewer::StatsHandler);

    appWindow.show();

    Fl::set_idle(idle_cb);

    return Fl::run();
}