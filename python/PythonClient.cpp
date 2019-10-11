#include <pybind11/pybind11.h>

#include <string>

#include "LbmFile.hpp"

class Client {
 private:
  LbmFile m_lbmFile;

 public:
  explicit Client(std::string lbmFilePath) : m_lbmFile(lbmFilePath) {}
  std::string getTitle() { return m_lbmFile.getTitle(); }
};

namespace py = pybind11;

PYBIND11_MODULE(python_lbm, m) {
  py::class_<Client>(m, "Client")
      .def(py::init<const std::string &>())
      .def("getTitle", &Client::getTitle);

  m.attr("__version__") = "dev";
}
