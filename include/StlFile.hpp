#pragma once

#include <string>
#include <vector>

#include "tao/pegtl.hpp"

namespace stl_file {

namespace pegtl = tao::pegtl;

class StlFile {
 public:
  std::string name;
  std::vector<float> normals;
  std::vector<float> vertices;

  explicit StlFile(std::string path);
};

}  // namespace stl_file

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

template <typename Out>
static void split(const std::string& s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

static std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

static void readVec(std::string in, std::vector<float>* out) {
  std::vector<std::string> floats = split(in, ' ');
  assert(floats.size() == 3);
  for (std::string fstring : floats) {
    out->push_back(std::stof(fstring));
  }
}

struct indent : pegtl::plus<pegtl::space> {};
struct opt_indent : pegtl::opt<indent> {};
struct name : pegtl::plus<pegtl::identifier_other> {};

struct double3
    : pegtl::seq<stl_double::grammar, pegtl::space, stl_double::grammar,
                 pegtl::space, stl_double::grammar> {};
struct normal_vec : double3 {};
struct vertex_vec : double3 {};

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

struct facet_grammar
    : pegtl::seq<facet_line, outerloop_line, pegtl::rep<3, vertex_line>,
                 endloop_line, endfacet_line> {};
struct grammar
    : pegtl::seq<solid_line, pegtl::until<endsolid_line,
                                          pegtl::rep_min<1, facet_grammar>>> {};

template <typename Rule>
struct action : pegtl::nothing<Rule> {};

template <>
struct action<name> {
  template <typename Input>
  static void apply(const Input& in, stl_file::StlFile* d) {
    d->name = in.string();
  }
};

template <>
struct action<normal_vec> {
  template <typename Input>
  static void apply(const Input& in, stl_file::StlFile* d) {
    readVec(in.string(), &d->normals);
  }
};

template <>
struct action<vertex_vec> {
  template <typename Input>
  static void apply(const Input& in, stl_file::StlFile* d) {
    readVec(in.string(), &d->vertices);
  }
};

}  // namespace stl_ascii
