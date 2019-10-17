#pragma once

#include <QFile>
#include <QFileDevice>
#include <QFileInfo>
#include <QTextStream>

#include <stdint.h>
#include <sys/time.h>
#include <string>
#include <vector>

#include "Average.hpp"
#include "Observer.hpp"

typedef Observer<const AverageMatrix&> AverageObserver;

class ListAveraging : public AverageObserver {
 private:
  AverageMatrix m_avgs;

 public:
  ListAveraging() : m_avgs() {}
  const AverageMatrix& getAverages() { return m_avgs; }
  void notify(const AverageMatrix& avgs) { m_avgs = avgs; }
};

/**
 * @brief Prints averaging data to stdout
 */
class StdoutAveraging : public AverageObserver {
 protected:
  QTextStream m_stream;
  bool m_isFirstWrite;

 public:
  void writeAverages(const AverageMatrix& avgMatrix) {
    AverageData avgs = avgMatrix.m_rows.back();
    uint64_t ticks = std::chrono::duration_cast<std::chrono::microseconds>(
                         avgs.m_time.time_since_epoch())
                         .count();
    // uint64_t ticks = avgs.m_time.tv_sec * 1000 + avgs.m_time.tv_usec / 1000;
    m_stream << ticks << ",";
    for (int i = 0; i < avgs.m_measurements.size(); i++) {
      Average avg = avgs.m_measurements.at(i);
      m_stream << avg.temperature << "," << avg.flow;
      if (i == avgs.m_measurements.size() - 1)
        m_stream << endl;
      else
        m_stream << ",";
    }
    m_stream.flush();
  }

  void writeHeaders(const AverageMatrix& avgs) {
    m_stream << "time,";
    for (int i = 0; i < avgs.m_columns.size(); i++) {
      QString name = QString::fromStdString(avgs.m_columns.at(i));
      m_stream << name << "_T," << name << "_Q";
      if (i == avgs.m_columns.size() - 1)
        m_stream << endl;
      else
        m_stream << ",";
    }
    m_stream.flush();
  }

  void notify(const AverageMatrix& avgs) {
    if (m_isFirstWrite) {
      writeHeaders(avgs);
      m_isFirstWrite = false;
    }
    writeAverages(avgs);
  }

  StdoutAveraging()
      : m_stream(stdout, QIODevice::WriteOnly), m_isFirstWrite(true) {}
};

/**
 * @brief Prints averaging data to a CSV file
 */
class CSVAveraging : public StdoutAveraging {
 private:
  QFile m_file;

 public:
  explicit CSVAveraging(std::string filePath)
      : StdoutAveraging(), m_file(QString::fromStdString(filePath)) {
    QFileInfo outputFileInfo(m_file);
    if (outputFileInfo.size() > 0) m_file.remove();
    if (m_file.open(QIODevice::WriteOnly | QIODevice::Append)) {
      m_stream.setDevice(&m_file);
    } else {
      throw std::runtime_error("Failed to open output CSV file");
    }
  }
};
