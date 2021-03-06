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
#include "BasicTimer.hpp"
#include "Observer.hpp"

typedef Observer<const AverageMatrix&> AverageObserver;

/**
 * @brief Observer for averaging values which creates a list in memory
 */
class ListAveraging : public AverageObserver {
 private:
  AverageMatrix m_avgs;

 public:
  ListAveraging() : m_avgs() {}
  const AverageMatrix& getAverages() { return m_avgs; }
  void notify(const AverageMatrix& avgs) { m_avgs = avgs; }
};

/**
 * @brief Observer for averaging values which prints data to stdout
 */
class StdoutAveraging : public AverageObserver {
 protected:
  QTextStream m_stream;
  bool m_isFirstWrite;

 public:
  void writeAverages(const AverageMatrix& avgMatrix) {
    AverageData avgs = avgMatrix.m_rows.back();
    // uint64_t ticks = std::chrono::duration_cast<std::chrono::microseconds>(
    //                      avgs.m_time.time_since_epoch())
    //                      .count();
    auto tpTime = sim_clock_t::to_time_t(avgs.m_time);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&tpTime), DATETIME_FMT) << ",";
    m_stream << QString::fromStdString(ss.str());
    for (size_t i = 0; i < avgs.m_measurements.size(); i++) {
      Average avg = avgs.m_measurements.at(i);
      m_stream << avg.temperature << "," << avg.flow;
      if (i == avgs.m_measurements.size() - 1)
        m_stream << "\n";
      else
        m_stream << ",";
    }
    m_stream.flush();
  }

  void writeHeaders(const AverageMatrix& avgs) {
    m_stream << "time,";
    for (size_t i = 0; i < avgs.m_columns.size(); i++) {
      QString name = QString::fromStdString(avgs.m_columns.at(i));
      m_stream << name << "_T," << name << "_Q";
      if (i == avgs.m_columns.size() - 1)
        m_stream << "\n";
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
