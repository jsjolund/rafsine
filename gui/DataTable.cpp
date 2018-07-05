#include "DataTable.hpp"

RealPointerInputBox::RealPointerInputBox(real *ptr,
                                         int X, int Y, int W, int H,
                                         const char *L = 0)
    : Fl_Float_Input(X, Y, W, H, L), ptr(ptr)
{
}

int RealPointerInputBox::handle(int event)
{
  if (event == FL_UNFOCUS)
  {
    real val = std::stof(std::string(value()));
    *ptr = val;
  }
  return Fl_Float_Input::handle(event);
}

LuaInputBox::LuaInputBox(std::string *ptr,
                         int X, int Y, int W, int H,
                         const char *L = 0)
    : Fl_Input(X, Y, W, H, L), ptr(ptr)
{
}

int LuaInputBox::handle(int event)
{
  if (event == FL_UNFOCUS)
  {
    std::string val = std::string(value());
    ptr->assign(val);
  }
  return Fl_Input::handle(event);
}

void DataTable::clear()
{
  tile->resize(x(), y(), w(), 0);
  widgets.clear();
  tile->clear();
}

void DataTable::setTableHeaders(const char *header[2], int numRows)
{
  rows = numRows;
  tile->resize(x(), y(), w(), cellh * rows);
  int col1w = w() - col0w;
  widgets.clear();
  tile->clear();
  tile->begin();
  {
    Fl_Box *box = new Fl_Box(x(), y(), col0w, cellh, header[0]);
    box->box(FL_UP_FRAME);
    widgets.push_back(box);
  }
  {
    Fl_Box *box = new Fl_Box(x() + col0w, y(), col1w, cellh, header[1]);
    box->box(FL_UP_FRAME);
    widgets.push_back(box);
  }
}

void DataTable::showFloatTable(const char *headers[2],
                               tsl::ordered_map<const char *, real *> *cmap)
{
  // A table for inputting floating point values to fixed variable names
  setTableHeaders(headers, cmap->size() + 1);
  // Create widgets
  int col1w = w() - col0w;
  int xx = x(), yy = y() + cellh;
  tsl::ordered_map<const char *, real *>::iterator it = cmap->begin();
  for (int r = 1; r < rows; r++)
  {
    for (int c = 0; c < cols; c++)
    {
      if (c == 0)
      {
        Fl_Box *box = new Fl_Box(xx, yy, col0w, cellh, it->first);
        box->box(FL_BORDER_BOX);
        widgets.push_back(box);
      }
      else
      {
        Fl_Float_Input *in = new RealPointerInputBox(it->second, xx, yy, col1w, cellh);
        in->box(FL_BORDER_BOX);
        std::string s = std::to_string(*(it->second));
        in->value(s.c_str());
        widgets.push_back(in);
      }
      xx += col0w;
    }
    xx = x();
    yy += cellh;
    ++it;
  }
  tile->end();
}

void DataTable::showStringTable(const char *headers[2],
                                tsl::ordered_map<std::string *, std::string *> *cmap)
{
  // A table for inputting string variable names and definitions
  setTableHeaders(headers, cmap->size() + 1);
  // Create widgets
  int col1w = w() - col0w;
  int xx = x(), yy = y() + cellh;
  tsl::ordered_map<std::string *, std::string *>::iterator it = cmap->begin();
  for (int r = 1; r < rows; r++)
  {
    for (int c = 0; c < cols; c++)
    {
      if (c == 0)
      {
        Fl_Input *box = new LuaInputBox(it->first, xx, yy, col0w, cellh);
        box->box(FL_BORDER_BOX);
        box->value(it->first->c_str());
        widgets.push_back(box);
      }
      else
      {
        Fl_Input *in = new LuaInputBox(it->second, xx, yy, col1w, cellh);
        in->box(FL_BORDER_BOX);
        in->value(it->second->c_str());
        widgets.push_back(in);
      }
      xx += col0w;
    }
    xx = x();
    yy += cellh;
    ++it;
  }
  tile->end();
}

void DataTable::showVoxelGeometryQuad(VoxelGeometryQuad *quad)
{
  const char *header[2] = {"Property", "Value"};
  tsl::ordered_map<const char *, real *> cmap;
  cmap["origin.x"] = &(quad->origin.x);
  cmap["origin.y"] = &(quad->origin.y);
  cmap["origin.z"] = &(quad->origin.z);
  cmap["dir1.x"] = &(quad->dir1.x);
  cmap["dir1.y"] = &(quad->dir1.y);
  cmap["dir1.z"] = &(quad->dir1.z);
  cmap["dir2.x"] = &(quad->dir2.x);
  cmap["dir2.y"] = &(quad->dir2.y);
  cmap["dir2.z"] = &(quad->dir2.z);

  // A table for inputting floating point values to fixed variable names
  setTableHeaders(header, cmap.size() + 1);
  // Create widgets
  int col1w = w() - col0w;
  int xx = x(), yy = y() + cellh;
  tsl::ordered_map<const char *, real *>::iterator it = cmap.begin();
  for (int r = 1; r < rows; r++)
  {
    for (int c = 0; c < cols; c++)
    {
      if (c == 0)
      {
        Fl_Box *box = new Fl_Box(xx, yy, col0w, cellh, it->first);
        box->box(FL_BORDER_BOX);
        widgets.push_back(box);
      }
      else
      {
        Fl_Float_Input *in = new Fl_Float_Input(xx, yy, col1w, cellh);
        in->box(FL_BORDER_BOX);
        std::string s = std::to_string(*(it->second));
        in->value(s.c_str());
        widgets.push_back(in);
      }
      xx += col0w;
    }
    xx = x();
    yy += cellh;
    ++it;
  }
  tile->end();
}

void DataTable::showSimConstants(SimConstants *c)
{
  const char *header[2] = {"Property", "Value"};
  tsl::ordered_map<const char *, real *> cmap;
  cmap["mx"] = &c->mx;
  cmap["my"] = &c->my;
  cmap["mz"] = &c->mz;
  cmap["nu"] = &c->nu;
  cmap["nuT"] = &c->nuT;
  cmap["C"] = &c->C;
  cmap["k"] = &c->k;
  cmap["Pr"] = &c->Pr;
  cmap["Pr_t"] = &c->Pr_t;
  cmap["gBetta"] = &c->gBetta;
  cmap["Tinit"] = &c->Tinit;
  cmap["Tref"] = &c->Tref;
  showFloatTable(header, &cmap);
}

void DataTable::showUnitConverter(UnitConverter *uc)
{
  const char *header[2] = {"Property", "Value"};
  tsl::ordered_map<const char *, real *> cmap;
  cmap["ref_L_phys"] = &uc->ref_L_phys;
  cmap["ref_L_lbm"] = &uc->ref_L_lbm;
  cmap["ref_U_phys"] = &uc->ref_U_phys;
  cmap["ref_U_lbm"] = &uc->ref_U_lbm;
  cmap["C_Temp"] = &uc->C_Temp;
  cmap["T0_phys"] = &uc->T0_phys;
  cmap["T0_lbm"] = &uc->T0_lbm;
  showFloatTable(header, &cmap);
}

void DataTable::showUserConstants(UserConstants *uc)
{
  const char *header[2] = {"Property", "Value"};
  setTableHeaders(header, uc->size() + 1);
  tsl::ordered_map<std::string *, std::string *> ptrMap;
  for (UserConstants::iterator it = uc->begin(); it != uc->end(); ++it)
  {
    // Is it safe to cast (const string*) to (string*)?
    std::string *k = (std::string *)&it->first;
    std::string *v = (std::string *)&it->second;
    ptrMap[k] = v;
  }
  showStringTable(header, &ptrMap);
}

void DataTable::resize(int X, int Y, int W, int H)
{
  int tileW = max(W, col0w * 2);
  int col1w = tileW - col0w;
  for (int r = 0; r < rows && (r * cols + 1) < widgets.size(); r++)
  {
    Fl_Widget *in = widgets.at(r * cols + 1);
    in->resize(in->x(), in->y(), col1w, in->h());
  }
  tile->resize(X, Y, tileW, cellh * rows);
  Fl_Scroll::resize(X, Y, W, H);
}

DataTable::DataTable(int X, int Y, int W, int H, const char *L)
    : Fl_Scroll(X, Y, W, H, L), cellh(25), col0w(120), rows(0), cols(2)
{
  tile = new Fl_Tile(X, Y, W, cellh * rows);
  tile->end();
  end();
}