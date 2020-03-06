#pragma once

#include <QByteArray>
#include <QCryptographicHash>
#include <QDataStream>
#include <QDir>
#include <QStandardPaths>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <LuaContext.hpp>

#include "LbmMethod.hpp"
#include "LuaGeometry.hpp"
#include "SimulationParams.hpp"
#include "UnitConverter.hpp"

/**
 * @brief Load data from lua code for a CFD problem
 *
 */
class LuaData {
 private:
  template <typename S, typename D>
  void readVariable(const std::string var, D* dst, LuaContext* lua);

  //! Hash for geometry (re)generation
  std::string m_hash;

 public:
  int m_nx, m_ny, m_nz;
  //! The real-to-lbm unit converter loaded from Lua
  std::shared_ptr<UnitConverter> m_unitConverter;
  //! Voxel/lattice geometry loaded from Lua script
  std::shared_ptr<VoxelGeometry> m_voxGeo;
  //! Some parameters for the CUDA kernel
  std::shared_ptr<SimulationParams> m_param;
  //! Averaging period
  float m_avgPeriod;
  //! Partitioning axis for multi-GPU
  D3Q4::Enum m_partitioning;
  //! LBM method
  LBM::Enum m_method;

  void loadSimulation(const std::string buildGeometryPath,
                      const std::string settingsPath);

  std::string getVoxelMeshPath() {
    if (m_hash.length() == 0) { return std::string(); }
    QString tmpPath =
        QStandardPaths::standardLocations(QStandardPaths::TempLocation).at(0);
    tmpPath.append(QDir::separator())
        .append("rafsine-")
        .append(QString(m_hash.c_str()))
        .append(".osgb");
    return tmpPath.toUtf8().constData();
  }
};
