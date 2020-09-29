#pragma once

#include <QDir>
#include <QFileInfo>
#include <QTextStream>
#include <stdexcept>
#include <string>

#include "BasicTimer.hpp"
#include "cpptoml.h"
#include "rapidcsv.h"

/**
 * @brief Class for reading .lbm files, defining paths for project files. Uses
 * the TOML syntax: https://github.com/toml-lang/toml
 *
 */
class LbmFile {
  std::string m_lbmFilePath;
  std::string m_settingsPath;
  std::string m_geometryPath;
  std::string m_inputCsvPath;
  std::string m_outputCsvPath;
  std::string m_author;
  std::string m_title;

 public:
  std::string getSettingsPath() { return m_settingsPath; }
  std::string getGeometryPath() { return m_geometryPath; }
  std::string getInputCSVPath() { return m_inputCsvPath; }
  std::string getOutputCSVPath() { return m_outputCsvPath; }
  std::string getAuthor() { return m_author; }
  std::string getTitle() { return m_title; }

  bool isValid() {
    return (m_settingsPath.length() > 0) && (m_geometryPath.length() > 0);
  }

  std::time_t getStartTime();

  LbmFile& operator=(const LbmFile& other) {
    m_lbmFilePath = other.m_lbmFilePath;
    m_settingsPath = other.m_settingsPath;
    m_geometryPath = other.m_geometryPath;
    m_inputCsvPath = other.m_inputCsvPath;
    m_outputCsvPath = other.m_outputCsvPath;
    m_author = other.m_author;
    m_title = other.m_title;
    return *this;
  }

  LbmFile() {}

  explicit LbmFile(QString lbmFilePath);

  explicit LbmFile(std::string lbmFilePath)
      : LbmFile(QString::fromStdString(lbmFilePath)) {}

  LbmFile(const LbmFile& other)
      : m_lbmFilePath(other.m_lbmFilePath),
        m_settingsPath(other.m_settingsPath),
        m_geometryPath(other.m_geometryPath),
        m_inputCsvPath(other.m_inputCsvPath),
        m_outputCsvPath(other.m_outputCsvPath),
        m_author(other.m_author),
        m_title(other.m_title) {}
};
