
#include <osg/ArgumentParser>
#include <osg/Vec4>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "pegtl.hh"

#include <boost/filesystem.hpp>

#include "box_triangle/aabb_triangle_overlap.h"
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"

std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec) {
  for (float s : vec) os << s << ", ";
  return os;
}

struct data {
  std::string name;
  std::vector<float> normals;
  std::vector<float> vertices;
};

namespace stldouble {
struct plus_minus : pegtl::opt<pegtl::one<'+', '-'>> {};
struct dot : pegtl::one<'.'> {};

struct inf : pegtl::seq<pegtl::istring<'i', 'n', 'f'>,
                        pegtl::opt<pegtl::istring<'i', 'n', 'i', 't', 'y'>>> {};

struct nan : pegtl::seq<pegtl::istring<'n', 'a', 'n'>,
                        pegtl::opt<pegtl::one<'('>, pegtl::plus<pegtl::alnum>,
                                   pegtl::one<')'>>> {};

template <typename D>
struct number
    : pegtl::if_then_else<
          dot, pegtl::plus<D>,
          pegtl::seq<pegtl::plus<D>, pegtl::opt<dot, pegtl::star<D>>>> {};

struct e : pegtl::one<'e', 'E'> {};
struct p : pegtl::one<'p', 'P'> {};
struct exponent : pegtl::seq<plus_minus, pegtl::plus<pegtl::digit>> {};

struct decimal : pegtl::seq<number<pegtl::digit>, pegtl::opt<e, exponent>> {};
struct binary : pegtl::seq<pegtl::one<'0'>, pegtl::one<'x', 'X'>,
                           number<pegtl::xdigit>, pegtl::opt<p, exponent>> {};

struct grammar : pegtl::seq<plus_minus, pegtl::sor<decimal, binary, inf, nan>> {
};

}  // namespace stldouble

namespace stlfile {

template <typename Out>
void split(const std::string& s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

struct indent : pegtl::plus<pegtl::space> {};
struct name : pegtl::plus<pegtl::identifier_other> {};

struct double3
    : pegtl::seq<stldouble::grammar, pegtl::space, stldouble::grammar,
                 pegtl::space, stldouble::grammar> {};
struct normal_vec : double3 {};
struct vertex_vec : double3 {};

struct vertex_str : pegtl::string<'v', 'e', 'r', 't', 'e', 'x'> {};
struct solid_str : pegtl::string<'s', 'o', 'l', 'i', 'd'> {};
struct outerloop_str
    : pegtl::string<'o', 'u', 't', 'e', 'r', ' ', 'l', 'o', 'o', 'p'> {};
struct endloop_str : pegtl::string<'e', 'n', 'd', 'l', 'o', 'o', 'p'> {};
struct endfacet_str : pegtl::string<'e', 'n', 'd', 'f', 'a', 'c', 'e', 't'> {};
struct endsolid_str : pegtl::string<'e', 'n', 'd', 's', 'o', 'l', 'i', 'd'> {};
struct facet_normal_str : pegtl::string<'f', 'a', 'c', 'e', 't', ' ', 'n', 'o',
                                        'r', 'm', 'a', 'l'> {};

struct solid_line : pegtl::must<solid_str, pegtl::space, pegtl::one<34>, name,
                                pegtl::one<34>, pegtl::eol> {};
struct facet_line : pegtl::seq<indent, facet_normal_str, pegtl::space,
                               normal_vec, pegtl::eol> {};
struct outerloop_line : pegtl::seq<indent, outerloop_str, pegtl::eol> {};
struct vertex_line
    : pegtl::seq<indent, vertex_str, pegtl::space, vertex_vec, pegtl::eol> {};
struct endloop_line : pegtl::seq<indent, endloop_str, pegtl::eol> {};
struct endfacet_line : pegtl::seq<indent, endfacet_str, pegtl::eol> {};
struct endsolid_line : pegtl::seq<endsolid_str, pegtl::eolf> {};

struct facet_grammar
    : pegtl::seq<facet_line, outerloop_line, pegtl::rep_min<3, vertex_line>,
                 endloop_line, endfacet_line> {};
struct grammar : pegtl::seq<solid_line, pegtl::rep_min<1, facet_grammar>,
                            pegtl::until<endsolid_line>> {};

template <typename Rule>
struct action : pegtl::nothing<Rule> {};

template <>
struct action<name> {
  static void apply(const pegtl::input& in, data& d) { d.name = in.string(); }
};

template <>
struct action<normal_vec> {
  static void apply(const pegtl::input& in, data& d) {
    std::vector<std::string> floats = split(in.string(), ' ');
    assert(floats.size() == 3);
    for (std::string fstring : floats) {
      d.normals.push_back(std::stof(fstring));
    }
  }
};

template <>
struct action<vertex_vec> {
  static void apply(const pegtl::input& in, data& d) {
    std::vector<std::string> floats = split(in.string(), ' ');
    assert(floats.size() == 3);
    for (std::string fstring : floats) {
      d.vertices.push_back(std::stof(fstring));
    }
  }
};
}  // namespace stlfile

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);

  std::string pathString;
  if (!args.read("-i", pathString)) {
    std::cout << "-i path/to/model.stl" << std::endl;
    return -1;
  }

  boost::filesystem::path input(pathString);
  std::vector<boost::filesystem::path> filePaths;
  boost::filesystem::directory_iterator end;
  for (boost::filesystem::directory_iterator it(input); it != end; ++it) {
    if (it->path().extension().string() == ".stl") {
      data solid;
      boost::filesystem::path filePath = it->path();
      pegtl::file_parser(filePath.string())
          .parse<stlfile::grammar, stlfile::action>(solid);
      std::cout << solid.name << std::endl;
      std::cout << solid.normals << std::endl;
      std::cout << solid.vertices << std::endl;
    }
  }

  // osg::ref_ptr<osg::Group> root = new osg::Group;
  // root->addChild(mesh);
  // osgViewer::Viewer viewer;
  // viewer.getCamera()->setClearColor(osg::Vec4(0, 0, 0, 1));
  // viewer.setSceneData(root);
  // viewer.setUpViewInWindow(400, 400, 800, 600);
  // viewer.addEventHandler(new osgViewer::StatsHandler);
  // return viewer.run();
  return 0;
}
