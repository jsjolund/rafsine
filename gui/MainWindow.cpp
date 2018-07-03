#include "MainWindow.hpp"

void MainWindow::setKernelData(KernelData *kernelData)
{
  tree->populate(kernelData);
}

void MainWindow::setVoxelMesh(VoxelMesh *mesh)
{
  mygl->setVoxelMesh(mesh);
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
                      int(float(w()) * 1 / 4 - resize_w),
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
  tree->table = table;
  end();
}