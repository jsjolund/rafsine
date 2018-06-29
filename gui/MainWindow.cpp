#include "MainWindow.hpp"

void MainWindow::setVoxelMesh(VoxelMesh *mesh)
{
  osg::Group *root = new osg::Group();
  osg::Geode *geode = new osg::Geode();
  osg::Geometry *polyGeom = new osg::Geometry();
  polyGeom->setVertexArray(mesh->vertices_);
  polyGeom->setColorArray(mesh->v_colors_);
  polyGeom->setNormalArray(mesh->normals_);
  polyGeom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
  polyGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
  int numVerts = mesh->vertices_->getNumElements();
  polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, numVerts));
  geode->addDrawable(polyGeom);
  // geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED);
  // geode->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
  // geode->getOrCreateStateSet()->setMode(GL_TEXTURE_2D, osg::StateAttribute::OFF);
  root->addChild(geode);

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
  root->addChild(lightGroup);

  mygl->setSceneData(root);
}

void MainWindow::resize(int X, int Y, int W, int H)
{
  Fl_Double_Window::resize(X, Y, W, H);

  menu->resize(0, 0, W, menu_h);
  statusBar->resize(0,
                    h() - statusBar_h,
                    w(),
                    statusBar_h);
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
                h() - (hbar->y() + hbar->h()) - statusBar_h);
  vbar->resize(tree->x() + tree->w(),
               menu_h,
               resize_w,
               h() - statusBar_h * 2);
  mygl->resize(vbar->x() + vbar->w(),
               menu_h,
               w() - tree->w(),
               h() - statusBar_h * 2);
}

MainWindow::MainWindow(int W, int H, const char *L) : Fl_Double_Window(W, H, L)
{
  menu = new MainMenu(0, 0, W, menu_h);
  statusBar = new StatusBar(0,
                            h() - statusBar_h,
                            w(),
                            statusBar_h);
  tree = new TreeView(0,
                      menu_h,
                      int(float(w()) * 1 / 6 - resize_w),
                      int(float(h()) * 1 / 2 - resize_h));
  hbar = new ResizerBarHoriz(0,
                             tree->y() + tree->h(),
                             tree->w(),
                             resize_h);
  table = new DataTable(0,
                        hbar->y() + hbar->h(),
                        hbar->w(),
                        h() - (hbar->y() + hbar->h()) - statusBar_h);
  vbar = new ResizerBarVert(tree->x() + tree->w(),
                            menu_h,
                            resize_w,
                            h() - statusBar_h);
  mygl = new GLWindow(vbar->x() + vbar->w(),
                      menu_h,
                      w() - tree->w(),
                      h() - statusBar_h);
  end();
}