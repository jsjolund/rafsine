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

#include "ResizeBar.hpp"
#include "GLWindow.hpp"
#include "TreeView.hpp"
#include "DataTable.hpp"

void Change_CB(Fl_Widget *w, void *)
{
    Fl_Menu_Bar *menu = (Fl_Menu_Bar *)w;
    Fl_Menu_Item *p;
    // Change submenu name
    p = (Fl_Menu_Item *)menu->find_item("Edit/Submenu");
    if (p)
        p->label("New Submenu Name");
    // Change item name
    p = (Fl_Menu_Item *)menu->find_item("Edit/New Submenu Name/Aaa");
    if (p)
        p->label("New Aaa Name");
}
void Quit_CB(Fl_Widget *, void *)
{
    exit(0);
}

// APP WINDOW CLASS
class MyAppWindow : public Fl_Double_Window
{
  public:
    GLWindow *mygl; // opengl window
  private:
    const int resize_h = 5;
    const int resize_w = 5;
    const int menu_h = 24;
    TreeView *tree;
    RateTable *table;
    ResizerBarHoriz *hbar;
    ResizerBarVert *vbar;
    Fl_Menu_Bar *menu;

  public:
    void resize(int X, int Y, int W, int H)
    {
        Fl_Double_Window::resize(X, Y, W, H);
        menu->resize(0,0,W,menu_h);
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
        menu = new Fl_Menu_Bar(0, 0, W, menu_h);
        menu->add("File/Quit", FL_CTRL + 'q', Quit_CB);
        menu->add("Edit/Change", FL_CTRL + 'c', Change_CB);
        menu->add("Edit/Submenu/Aaa");
        menu->add("Edit/Submenu/Bbb");
        std::cout << "menu" << std::endl;
        tree = new TreeView(0,
                            menu_h,
                            int(float(w()) * 1 / 3 - resize_w),
                            int(float(h()) * 1 / 2 - resize_h));
        hbar = new ResizerBarHoriz(0,
                                   tree->y() + tree->h(),
                                   tree->w(),
                                   resize_h);
        table = new RateTable(0,
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

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << argv[0] << ": requires filename argument." << std::endl;
        return 1;
    }
    osg::ArgumentParser arguments(&argc, argv);

    // load the scene.
    osg::ref_ptr<osg::Node> loadedModel = osgDB::readRefNodeFiles(arguments);
    if (!loadedModel)
    {
        std::cout << argv[0] << ": No data loaded." << std::endl;
        return 1;
    }

    MyAppWindow appWindow(1024, 800, "My app");
    appWindow.resizable(&appWindow);

    appWindow.mygl->setSceneData(loadedModel.get());
    appWindow.mygl->setCameraManipulator(new osgGA::TrackballManipulator);
    appWindow.mygl->addEventHandler(new osgViewer::StatsHandler);

    appWindow.show();

    Fl::set_idle(idle_cb);

    return Fl::run();
}