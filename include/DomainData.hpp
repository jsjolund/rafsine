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

#include "KernelExecutor.hpp"
#include "KernelInterface.hpp"
#include "LBM_Algo.hpp"
#include "LuaGeometry.hpp"
#include "SimulationParams.hpp"
#include "SimulationTimer.hpp"
#include "UnitConverter.hpp"

/**
 * @brief Load data from lua code for a CFD problem
 *
 */
class DomainData {
 private:
  template <typename S, typename D>
  void readVariable(const std::string var, D* dst, LuaContext* lua);

  //! Hash for geometry (re)generation
  std::string m_hash;

  /**
   * @brief Create hash for geometry caching and (re)generation. Depends on
   * content of input parameters.
   *
   * @param buildGeometryPath LUA script to build geometry. Full content of file
   * is used.
   * @param partitioning Partitioning axis
   * @param nx Number of lattice sites along X-axis
   * @param ny Number of lattice sites along X-axis
   * @param nz Number of lattice sites along X-axis
   * @return std::string Cryptographic hash string
   */
  std::string createGeometryHash(const std::string buildGeometryPath,
                                 const std::string partitioning,
                                 const size_t nx,
                                 const size_t ny,
                                 const size_t nz);

 public:
  //! Lattice size X-axis
  size_t m_nx;
  //! Lattice size Y-axis
  size_t m_ny;
  //! Lattice size Z-axis
  size_t m_nz;
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

  //! Interface to CUDA kernel
  std::shared_ptr<KernelInterface> m_kernel;
  //! An ordered list of boundary condition details
  std::shared_ptr<BoundaryConditions> m_bcs;
  //! Areas on which to perform temperature and velocity averaging
  std::shared_ptr<VoxelCuboidArray> m_avgs;
  //! Timer counting time passed in the simulation
  std::shared_ptr<SimulationTimer> m_timer;

  /**
   * @brief Load a simulation scenario from LUA code
   *
   * @param buildGeometryPath geometry.lua
   * @param settingsPath settings.lua
   */
  void loadSimulation(size_t nd,
                      const std::string buildGeometryPath,
                      const std::string settingsPath);

  /**
   * @brief Get the path of cached geometry based on cryptographic
   * hash.
   *
   * @return std::string
   */
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
