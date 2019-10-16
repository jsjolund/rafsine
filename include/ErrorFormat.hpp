#pragma once

#include <iostream>
#include <sstream>
#include <string>

/**
 * @brief A class for handling runtime errors
 *
 */
class ErrorFormat {
 public:
  ErrorFormat() {}
  ~ErrorFormat() {}

  /**
   * @brief Output operator
   *
   * @tparam Type
   * @param value
   * @return ErrorFormat&
   */
  template <typename Type>
  ErrorFormat& operator<<(const Type& value) {
    m_stream << value;
    return *this;
  }
  /**
   * @brief Error message as a string
   *
   * @return std::string
   */
  std::string str() const { return m_stream.str(); }
  /**
   * @brief Error message as a string
   *
   * @return std::string
   */
  operator std::string() const { return m_stream.str(); }

  /**
   * @brief Description
   *
   */
  enum ConvertToString { to_str };
  /**
   * @brief Input operator
   *
   * @return std::string
   */
  std::string operator>>(ConvertToString) { return m_stream.str(); }

 private:
  std::stringstream m_stream;

  ErrorFormat(const ErrorFormat&);
  ErrorFormat& operator=(ErrorFormat&);
};
