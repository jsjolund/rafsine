#include "StlFile.hpp"

stl_file::StlFile::StlFile(std::string path) {
  pegtl::file_input<> in(path);
  tao::pegtl::parse<stl_ascii::grammar, stl_ascii::action>(in, this);
  assert(normals.size() * 3 == vertices.size());
}
