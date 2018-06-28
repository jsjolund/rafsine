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
  mygl->setSceneData(root);
}
 
void MainWindow::resize(int X, int Y, int W, int H)
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

MainWindow::MainWindow(int W, int H, const char *L) : Fl_Double_Window(W, H, L)
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