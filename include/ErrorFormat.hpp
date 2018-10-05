#pragma once

#include <iostream>
#include <sstream>
#include <string>

class ErrorFormat {
 public:
  ErrorFormat() {}
  ~ErrorFormat() {}

  template <typename Type>
  ErrorFormat &operator<<(const Type &value) {
    m_stream << value;
    return *this;
  }

  std::string str() const { return m_stream.str(); }
  operator std::string() const { return m_stream.str(); }

  enum ConvertToString { to_str };
  std::string operator>>(ConvertToString) { return m_stream.str(); }

 private:
  std::stringstream m_stream;

  ErrorFormat(const ErrorFormat &);
  ErrorFormat &operator=(ErrorFormat &);
};
