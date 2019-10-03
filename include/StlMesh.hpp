#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "tao/pegtl.hpp"

/**
 * @brief Defines the PEGTL rules for parsing a double from a string
 *
 */
namespace stl_double {

namespace pegtl = tao::pegtl;

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

}  // namespace stl_double

/**
 * @brief Defines the PEGTL rules for parsing an ascii STL model file into
 * vectors of triangle normals and their vertices.
 *
 */
namespace stl_ascii {

namespace pegtl = tao::pegtl;

struct indent : pegtl::plus<pegtl::space> {};
struct opt_indent : pegtl::opt<indent> {};
struct name : pegtl::plus<pegtl::identifier_other> {};

struct normal_float : stl_double::grammar {};
struct vertex_float : stl_double::grammar {};

struct normal_vec : pegtl::seq<normal_float, pegtl::space, normal_float,
                               pegtl::space, normal_float> {};
struct vertex_vec : pegtl::seq<vertex_float, pegtl::space, vertex_float,
                               pegtl::space, vertex_float> {};

struct vertex_l : pegtl::string<'v', 'e', 'r', 't', 'e', 'x'> {};
struct solid_l : pegtl::string<'s', 'o', 'l', 'i', 'd'> {};
struct outerloop_l
    : pegtl::string<'o', 'u', 't', 'e', 'r', ' ', 'l', 'o', 'o', 'p'> {};
struct endloop_l : pegtl::string<'e', 'n', 'd', 'l', 'o', 'o', 'p'> {};
struct endfacet_l : pegtl::string<'e', 'n', 'd', 'f', 'a', 'c', 'e', 't'> {};
struct endsolid_l : pegtl::string<'e', 'n', 'd', 's', 'o', 'l', 'i', 'd'> {};
struct facet_normal_l : pegtl::string<'f', 'a', 'c', 'e', 't', ' ', 'n', 'o',
                                      'r', 'm', 'a', 'l'> {};

struct vertex_u : pegtl::string<'V', 'E', 'R', 'T', 'E', 'X'> {};
struct solid_u : pegtl::string<'S', 'O', 'L', 'I', 'D'> {};
struct outerloop_u
    : pegtl::string<'O', 'U', 'T', 'E', 'R', ' ', 'L', 'O', 'O', 'P'> {};
struct endloop_u : pegtl::string<'E', 'N', 'D', 'L', 'O', 'O', 'P'> {};
struct endfacet_u : pegtl::string<'E', 'N', 'D', 'F', 'A', 'C', 'E', 'T'> {};
struct endsolid_u : pegtl::string<'E', 'N', 'D', 'S', 'O', 'L', 'I', 'D'> {};
struct facet_normal_u : pegtl::string<'F', 'A', 'C', 'E', 'T', ' ', 'N', 'O',
                                      'R', 'M', 'A', 'L'> {};

struct vertex_str : pegtl::sor<vertex_l, vertex_u> {};
struct solid_str : pegtl::sor<solid_l, solid_u> {};
struct outerloop_str : pegtl::sor<outerloop_l, outerloop_u> {};
struct endloop_str : pegtl::sor<endloop_l, endloop_u> {};
struct endfacet_str : pegtl::sor<endfacet_l, endfacet_u> {};
struct endsolid_str : pegtl::sor<endsolid_l, endsolid_u> {};
struct facet_normal_str : pegtl::sor<facet_normal_l, facet_normal_u> {};

struct opt_quote : pegtl::opt<pegtl::one<34>> {};
struct opt_name : pegtl::seq<opt_quote, pegtl::opt<name>, opt_quote> {};

struct solid_line : pegtl::must<solid_str, pegtl::space, opt_name, pegtl::eol> {
};

struct facet_line : pegtl::seq<opt_indent, facet_normal_str, pegtl::space,
                               normal_vec, pegtl::eol> {};
struct outerloop_line : pegtl::seq<opt_indent, outerloop_str, pegtl::eol> {};
struct vertex_line
    : pegtl::seq<opt_indent, vertex_str, indent, vertex_vec, pegtl::eol> {};
struct endloop_line : pegtl::seq<opt_indent, endloop_str, pegtl::eol> {};
struct endfacet_line : pegtl::seq<opt_indent, endfacet_str, pegtl::eol> {};
struct endsolid_line
    : pegtl::seq<endsolid_str, pegtl::opt<pegtl::plus<pegtl::any>>,
                 pegtl::eolf> {};

struct facet_array
    : pegtl::seq<facet_line, outerloop_line, pegtl::rep<3, vertex_line>,
                 endloop_line, endfacet_line> {};
struct grammar
    : pegtl::seq<solid_line,
                 pegtl::until<endsolid_line, pegtl::rep_min<1, facet_array>>> {
};

template <typename Rule>
struct action : pegtl::nothing<Rule> {};

}  // namespace stl_ascii

namespace stl_binary {

namespace pegtl = tao::pegtl;

struct newline : pegtl::uint8::one<0x0A> {};
struct ascii : pegtl::uint8::range<0x20, 0x7E> {};

struct name : pegtl::must<pegtl::rep<80, pegtl::sor<newline, ascii>>> {};
struct tri_count : pegtl::must<pegtl::rep<1, pegtl::uint32_le::any>> {};

struct normal_float : pegtl::uint32_le::any {};
struct vertex_float : pegtl::uint32_le::any {};

struct normal_vec : pegtl::must<pegtl::rep<3, normal_float>> {};
struct vertex_vec : pegtl::must<pegtl::rep<3, vertex_float>> {};

struct facet : pegtl::seq<normal_vec, pegtl::rep<3, vertex_vec>> {};
struct facet_array : pegtl::rep_min<1, facet> {};

struct attribute_byte_count
    : pegtl::must<pegtl::rep<1, pegtl::uint16_le::any>> {};

struct grammar : pegtl::seq<name, tri_count, facet_array, attribute_byte_count,
                            pegtl::eolf> {};
template <typename Rule>
struct action : pegtl::nothing<Rule> {};

}  // namespace stl_binary

/**
 * @brief A 3D model with vertices and normals as floating point vectors
 *
 */
namespace stl_mesh {

namespace pegtl = tao::pegtl;

class StlMesh {
 public:
  std::string name;
  std::vector<float> normals;
  std::vector<float> vertices;

  explicit StlMesh(std::string path) {
    pegtl::file_input<> in(path);
    tao::pegtl::parse<stl_ascii::grammar, stl_ascii::action>(in, this);
    // tao::pegtl::parse<stl_binary::grammar, stl_binary::action>(in, this);
    assert(normals.size() * 3 == vertices.size());
  }
};

}  // namespace stl_mesh

namespace stl_ascii {

template <>
struct action<name> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->name = in.string();
  }
};

template <>
struct action<normal_float> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.push_back(std::stof(in.string()));
  }
};

template <>
struct action<vertex_float> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.push_back(std::stof(in.string()));
  }
};

}  // namespace stl_ascii

namespace stl_binary {

template <>
struct action<name> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->name = in.string();
  }
};

template <>
struct action<tri_count> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    std::cout << *((int*)in.begin()) << std::endl;
  }
};

template <>
struct action<normal_float> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    std::cout << "n " << *((float*)in.begin()) << std::endl;
  }
};

template <>
struct action<vertex_float> {
  template <typename Input>
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    std::cout << "v " << *((float*)in.begin()) << std::endl;
  }
};

}  // namespace stl_binary