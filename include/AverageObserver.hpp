#pragma once

#include <QFile>
#include <QFileDevice>
#include <QFileInfo>
#include <QTextStream>

#include <sys/time.h>
#include <string>
#include <vector>

#include "Average.hpp"
#include "Observer.hpp"

typedef Observer<const AverageData&> AverageObserver;

class StdoutObserver : public AverageObserver {
 protected:
  QTextStream m_stream;
  bool m_isFirstWrite;

 public:
  void writeAverages(const AverageData& avgs) {
    long int ticks = avgs.time.tv_sec * 1000 + avgs.time.tv_usec / 1000;
    m_stream << ticks << ",";
    for (int i = 0; i < avgs.rows.size(); i++) {
      Average avg = avgs.rows.at(i);
      m_stream << avg.m_temperature << "," << avg.m_flow;
      if (i == avgs.rows.size() - 1)
        m_stream << endl;
      else
        m_stream << ",";
    }
    m_stream.flush();
  }

  void writeHeaders(const AverageData& avgs) {
    m_stream << "time,";
    for (int i = 0; i < avgs.rows.size(); i++) {
      QString name = QString::fromStdString(avgs.rows.at(i).m_volume.getName());
      m_stream << name << "_T," << name << "_Q";
      if (i == avgs.rows.size() - 1)
        m_stream << endl;
      else
        m_stream << ",";
    }
    m_stream.flush();
  }

  void notify(const AverageData& avgs) {
    if (m_isFirstWrite) {
      writeHeaders(avgs);
      m_isFirstWrite = false;
    }
    writeAverages(avgs);
  }

  StdoutObserver()
      : m_stream(stdout, QIODevice::WriteOnly), m_isFirstWrite(true) {}
};

class CSVFileObserver : public StdoutObserver {
 private:
  QFile m_file;

 public:
  explicit CSVFileObserver(std::string filePath)
      : StdoutObserver(), m_file(QString::fromStdString(filePath)) {
    QFileInfo outputFileInfo(m_file);
    if (outputFileInfo.size() > 0) m_file.remove();
    if (m_file.open(QIODevice::WriteOnly | QIODevice::Append)) {
      m_stream.setDevice(&m_file);
    } else {
      throw std::runtime_error("Failed to open output CSV file");
    }
  }
};
