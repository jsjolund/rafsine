#pragma once

#include <regex>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "tao/pegtl.hpp"

/**
 * @brief Defines the PEGTL rules for parsing a double from a string
 */
namespace stl_double {

namespace pegtl = tao::pegtl;

struct plus_minus : pegtl::opt<pegtl::one<'+', '-'>> {};
struct dot : pegtl::one<'.'> {};

struct inf : pegtl::seq<pegtl::istring<'i', 'n', 'f'>,
                        pegtl::opt<pegtl::istring<'i', 'n', 'i', 't', 'y'>>> {};

struct nan : pegtl::seq<pegtl::istring<'n', 'a', 'n'>,
                        pegtl::opt<pegtl::one<'('>,
                                   pegtl::plus<pegtl::alnum>,
                                   pegtl::one<')'>>> {};

template <typename D>
struct number
    : pegtl::if_then_else<
          dot,
          pegtl::plus<D>,
          pegtl::seq<pegtl::plus<D>, pegtl::opt<dot, pegtl::star<D>>>> {};

struct e : pegtl::one<'e', 'E'> {};
struct p : pegtl::one<'p', 'P'> {};
struct exponent : pegtl::seq<plus_minus, pegtl::plus<pegtl::digit>> {};

struct decimal : pegtl::seq<number<pegtl::digit>, pegtl::opt<e, exponent>> {};
struct binary : pegtl::seq<pegtl::one<'0'>,
                           pegtl::one<'x', 'X'>,
                           number<pegtl::xdigit>,
                           pegtl::opt<p, exponent>> {};

struct grammar : pegtl::seq<plus_minus, pegtl::sor<decimal, binary, inf, nan>> {
};

}  // namespace stl_double

/**
 * @brief Defines the PEGTL rules for parsing an ASCII STL model file into
 * vectors of triangle normals and their vertices.
 */
namespace stl_ascii {

namespace pegtl = tao::pegtl;

struct indent : pegtl::plus<pegtl::space> {};
struct opt_indent : pegtl::opt<indent> {};
struct name : pegtl::plus<pegtl::identifier_other> {};

struct normal_float_x : stl_double::grammar {};
struct normal_float_y : stl_double::grammar {};
struct normal_float_z : stl_double::grammar {};
struct vertex_float_x : stl_double::grammar {};
struct vertex_float_y : stl_double::grammar {};
struct vertex_float_z : stl_double::grammar {};

struct normal_vec : pegtl::seq<normal_float_x,
                               pegtl::space,
                               normal_float_y,
                               pegtl::space,
                               normal_float_z> {};
struct vertex_vec : pegtl::seq<vertex_float_x,
                               pegtl::space,
                               vertex_float_y,
                               pegtl::space,
                               vertex_float_z> {};

struct vertex_l : pegtl::string<'v', 'e', 'r', 't', 'e', 'x'> {};
struct solid_l : pegtl::string<'s', 'o', 'l', 'i', 'd'> {};
struct outerloop_l
    : pegtl::string<'o', 'u', 't', 'e', 'r', ' ', 'l', 'o', 'o', 'p'> {};
struct endloop_l : pegtl::string<'e', 'n', 'd', 'l', 'o', 'o', 'p'> {};
struct endfacet_l : pegtl::string<'e', 'n', 'd', 'f', 'a', 'c', 'e', 't'> {};
struct endsolid_l : pegtl::string<'e', 'n', 'd', 's', 'o', 'l', 'i', 'd'> {};
struct facet_normal_l
    : pegtl::
          string<'f', 'a', 'c', 'e', 't', ' ', 'n', 'o', 'r', 'm', 'a', 'l'> {};

struct vertex_u : pegtl::string<'V', 'E', 'R', 'T', 'E', 'X'> {};
struct solid_u : pegtl::string<'S', 'O', 'L', 'I', 'D'> {};
struct outerloop_u
    : pegtl::string<'O', 'U', 'T', 'E', 'R', ' ', 'L', 'O', 'O', 'P'> {};
struct endloop_u : pegtl::string<'E', 'N', 'D', 'L', 'O', 'O', 'P'> {};
struct endfacet_u : pegtl::string<'E', 'N', 'D', 'F', 'A', 'C', 'E', 'T'> {};
struct endsolid_u : pegtl::string<'E', 'N', 'D', 'S', 'O', 'L', 'I', 'D'> {};
struct facet_normal_u
    : pegtl::
          string<'F', 'A', 'C', 'E', 'T', ' ', 'N', 'O', 'R', 'M', 'A', 'L'> {};

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

struct facet_line : pegtl::seq<opt_indent,
                               facet_normal_str,
                               pegtl::space,
                               normal_vec,
                               pegtl::eol> {};
struct outerloop_line : pegtl::seq<opt_indent, outerloop_str, pegtl::eol> {};
struct vertex_line
    : pegtl::seq<opt_indent, vertex_str, indent, vertex_vec, pegtl::eol> {};
struct endloop_line : pegtl::seq<opt_indent, endloop_str, pegtl::eol> {};
struct endfacet_line : pegtl::seq<opt_indent, endfacet_str, pegtl::eol> {};
struct endsolid_line : pegtl::seq<endsolid_str,
                                  pegtl::opt<pegtl::plus<pegtl::any>>,
                                  pegtl::eolf> {};

struct facet_array : pegtl::seq<facet_line,
                                outerloop_line,
                                pegtl::rep<3, vertex_line>,
                                endloop_line,
                                endfacet_line> {};
struct grammar
    : pegtl::must<solid_line, pegtl::until<endsolid_line, facet_array>> {};

template <typename Rule>
struct action : pegtl::nothing<Rule> {};

}  // namespace stl_ascii

/**
 * @brief Defines the PEGTL rules for parsing a binary STL model file into
 * vectors of triangle normals and their vertices.
 */
namespace stl_binary {

namespace pegtl = tao::pegtl;

struct name : pegtl::must<pegtl::rep<80, pegtl::uint8::any>> {};
struct tri_count : pegtl::must<pegtl::rep<1, pegtl::uint32_le::any>> {};

struct normal_float_x : pegtl::uint32_le::any {};
struct normal_float_y : pegtl::uint32_le::any {};
struct normal_float_z : pegtl::uint32_le::any {};
struct vertex_float_x : pegtl::uint32_le::any {};
struct vertex_float_y : pegtl::uint32_le::any {};
struct vertex_float_z : pegtl::uint32_le::any {};

struct normal_vec
    : pegtl::must<pegtl::seq<normal_float_x, normal_float_y, normal_float_z>> {
};
struct vertex_vec
    : pegtl::must<pegtl::seq<vertex_float_x, vertex_float_y, vertex_float_z>> {
};

struct attribute_byte_count
    : pegtl::must<pegtl::rep<1, pegtl::uint16_le::any>> {};

struct facet_array
    : pegtl::seq<normal_vec, pegtl::rep<3, vertex_vec>, attribute_byte_count> {
};

struct grammar
    : pegtl::must<name, tri_count, pegtl::until<pegtl::eof, facet_array>> {};

template <typename Rule>
struct action : pegtl::nothing<Rule> {};

}  // namespace stl_binary

/**
 * @brief Constructs a 3D model composed of triangles with vertices and normals
 * as floating point vectors
 */
namespace stl_mesh {

namespace pegtl = tao::pegtl;

class StlMesh {
 public:
  std::string name;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3f> vertices;

  explicit StlMesh(std::string path) {
    pegtl::file_input<> in(path);
    try {
      tao::pegtl::parse<stl_ascii::grammar, stl_ascii::action>(in, this);
    } catch (const tao::pegtl::parse_error& ae) {
      normals.clear();
      vertices.clear();

      try {
        tao::pegtl::parse<stl_binary::grammar, stl_binary::action>(in, this);
      } catch (const tao::pegtl::parse_error& be) {
        // File is neither ASCII nor binary STL
        std::stringstream msg;
        msg << "ASCII STL error:" << std::string(ae.what()) << ", "
            << "BINARY STL error:" << std::string(be.what());
        throw std::runtime_error(msg.str());
      }
    }
  }
};

}  // namespace stl_mesh

namespace stl_ascii {

template <>
struct action<name> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::name
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->name = in.string();
  }
};

template <>
struct action<normal_float_x> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::normal_float_x
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.push_back(Eigen::Vector3f());
    d->normals.back().x() = std::stof(in.string());
  }
};

template <>
struct action<normal_float_y> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::normal_float_y
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.back().y() = std::stof(in.string());
  }
};

template <>
struct action<normal_float_z> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::normal_float_z
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.back().z() = std::stof(in.string());
  }
};

template <>
struct action<vertex_float_x> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::vertex_float_x
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.push_back(Eigen::Vector3f());
    d->vertices.back().x() = std::stof(in.string());
  }
};

template <>
struct action<vertex_float_y> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::vertex_float_y
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.back().y() = std::stof(in.string());
  }
};

template <>
struct action<vertex_float_z> {
  template <typename Input>
  /**
   * @param in ASCII characters parsed by stl_ascii::vertex_float_z
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.back().z() = std::stof(in.string());
  }
};

}  // namespace stl_ascii

namespace stl_binary {

template <>
struct action<name> {
  template <typename Input>
  /**
   * @param in 80 ASCII characters parsed by stl_binary::name
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->name =
        std::regex_replace(std::string(in.begin(), 80), std::regex(" +$"), "");
  }
};

template <>
struct action<normal_float_x> {
  template <typename Input>
  /**
   * @param in 4 bytes floating point value parsed by stl_binary::normal_float_x
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.push_back(Eigen::Vector3f());
    d->normals.back().x() = *reinterpret_cast<const float*>(in.begin());
  }
};

template <>
struct action<normal_float_y> {
  template <typename Input>
  /**
   * @param in 4 bytes floating point value parsed by stl_binary::normal_float_y
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.back().y() = *reinterpret_cast<const float*>(in.begin());
  }
};

template <>
struct action<normal_float_z> {
  template <typename Input>
  /**
   * @param in 4 bytes floating point value parsed by stl_binary::normal_float_z
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->normals.back().z() = *reinterpret_cast<const float*>(in.begin());
  }
};

template <>
struct action<vertex_float_x> {
  template <typename Input>
  /**
   * @param in 4 bytes floating point value parsed by stl_binary::vertex_float_x
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.push_back(Eigen::Vector3f());
    d->vertices.back().x() = *reinterpret_cast<const float*>(in.begin());
  }
};

template <>
struct action<vertex_float_y> {
  template <typename Input>
  /**
   * @param in 4 bytes floating point value parsed by stl_binary::vertex_float_y
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.back().y() = *reinterpret_cast<const float*>(in.begin());
  }
};

template <>
struct action<vertex_float_z> {
  template <typename Input>
  /**
   * @param in 4 bytes floating point value parsed by stl_binary::vertex_float_z
   * @param d The mesh
   */
  static void apply(const Input& in, stl_mesh::StlMesh* d) {
    d->vertices.back().z() = *reinterpret_cast<const float*>(in.begin());
  }
};

}  // namespace stl_binary
