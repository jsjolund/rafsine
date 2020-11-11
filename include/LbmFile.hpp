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
 private:
  std::string m_lbmFilePath;
  std::string m_settingsPath;
  std::string m_geometryPath;
  std::string m_inputCsvPath;
  std::string m_outputCsvPath;
  std::string m_author;
  std::string m_title;

 public:
  /**
   * @return std::string Path to settings.lua
   */
  std::string getSettingsPath() { return m_settingsPath; }
  /**
   * @return std::string Path to geometry.lua
   */
  std::string getGeometryPath() { return m_geometryPath; }
  /**
   * @return std::string Path to simulation boundary condition input CSV file
   */
  std::string getInputCSVPath() { return m_inputCsvPath; }
  /**
   * @return std::string Path to write time averaged measurements to CSV file
   */
  std::string getOutputCSVPath() { return m_outputCsvPath; }
  /**
   * @return std::string Author name from project.lbm
   */
  std::string getAuthor() { return m_author; }
  /**
   * @return std::string Scenario title from project.lbm
   */
  std::string getTitle() { return m_title; }
  /**
   * @return true If settings.lua and geometry.lua are valid paths
   * @return false
   */
  bool isValid() {
    return (m_settingsPath.length() > 0) && (m_geometryPath.length() > 0);
  }
  /**
   * @brief Get the simulation start time (from input CSV)
   *
   * @return std::time_t
   */
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
