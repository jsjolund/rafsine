#include "LbmFile.hpp"

timeval LbmFile::getStartTime() {
  timeval startTime{.tv_sec = 0, .tv_usec = 0};
  if (m_inputCsvPath.length() == 0) return startTime;
  rapidcsv::Document doc(m_inputCsvPath, rapidcsv::LabelParams(0, -1));
  uint64_t t0 = doc.GetCell<uint64_t>(0, 0);
  startTime.tv_sec = t0;
  return startTime;
}

LbmFile::LbmFile(QString lbmFilePath) {
  m_lbmFilePath = lbmFilePath.toUtf8().constData();
  QFileInfo lbmFileInfo(lbmFilePath);

  if (!lbmFileInfo.isReadable()) {
    throw std::invalid_argument("LBM file not readable.");

  } else {
    QDir lbmFileParentDir = lbmFileInfo.dir();
    std::cout << "Parsing file " << m_lbmFilePath << std::endl;
    auto lbmFile = cpptoml::parse_file(m_lbmFilePath);
    // Required settings Lua file
    auto settingsPathPtr = lbmFile->get_as<std::string>("settings_lua");
    if (settingsPathPtr) {
      QString settingsPath =
          lbmFileParentDir.filePath(QString((*settingsPathPtr).c_str()));
      QFileInfo settingsFileInfo(settingsPath);
      if (settingsFileInfo.isReadable()) {
        m_settingsPath = settingsPath.toUtf8().constData();
      } else {
        throw std::invalid_argument("Lua settings file not readable.");
      }
    } else {
      throw std::invalid_argument("Lua settings file not defined.");
    }
    // Required geometry Lua file
    auto geometryPathPtr = lbmFile->get_as<std::string>("geometry_lua");
    if (geometryPathPtr) {
      QString geometryPath =
          lbmFileParentDir.filePath(QString((*geometryPathPtr).c_str()));
      QFileInfo geometryFileInfo(geometryPath);
      if (geometryFileInfo.isReadable()) {
        m_geometryPath = geometryPath.toUtf8().constData();
      } else {
        throw std::invalid_argument("Lua geometry file not readable.");
      }
    } else {
      throw std::invalid_argument("Lua geometry file not defined.");
    }
    // Optional input CSV, should be commented out if not present
    auto inputCsvPathPtr = lbmFile->get_as<std::string>("input_csv");
    if (inputCsvPathPtr) {
      QString inputCsvPath =
          lbmFileParentDir.filePath(QString((*inputCsvPathPtr).c_str()));
      QFileInfo inputCsvFileInfo(inputCsvPath);
      if (inputCsvFileInfo.isReadable()) {
        m_inputCsvPath = inputCsvPath.toUtf8().constData();
      } else {
        throw std::invalid_argument("Input CSV file not readable.");
      }
    } else {
      m_inputCsvPath = "";
    }
    // Optional output CSV, should be writable if defined
    auto outputCsvPathPtr = lbmFile->get_as<std::string>("output_csv");
    if (outputCsvPathPtr) {
      QString outputCsvPath =
          lbmFileParentDir.filePath(QString((*outputCsvPathPtr).c_str()));
      m_outputCsvPath = outputCsvPath.toUtf8().constData();
      QFileInfo outputCsvFileInfo(outputCsvPath);
      if (outputCsvFileInfo.exists()) {
        if (outputCsvFileInfo.isWritable()) {
          // Delete the old file
          QFile file(outputCsvPath);
          file.remove();
        } else {
          throw std::invalid_argument("Output CSV file not writable.");
        }
      }
      // Create a new empty output CSV file
      QFile file(outputCsvPath);
      file.open(QIODevice::WriteOnly | QIODevice::Text);
      QTextStream stream(&file);
      stream << "";
      file.close();
    } else {
      m_outputCsvPath = "";
    }
    // LBM problem title
    auto titlePtr = lbmFile->get_as<std::string>("title");
    if (titlePtr) {
      m_title = *titlePtr;
    } else {
      m_title = "Untitled";
    }
    // Project author
    auto authorPtr = lbmFile->get_as<std::string>("author");
    if (authorPtr) {
      m_author = *authorPtr;
    } else {
      m_author = "Unknown author";
    }
  }
}