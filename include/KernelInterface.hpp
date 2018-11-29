#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_profiler_api.h>

#include <omp.h>

#include <vector>

#include "BoundaryCondition.hpp"
#include "DistributionFunction.hpp"
#include "Kernel.hpp"

using thrust::device_vector;

class DeviceParams {
 public:
  std::vector<bool> peerAccessList;  //!< List of P2P access enabled
  cudaStream_t computeStream;        //!< LBM compute stream
  cudaStream_t dfExchangeStream;     //!< Velocity df halo exchange stream
  cudaStream_t dfTExchangeStream;    //!< Temperature df halo exchange stream
  DeviceParams(int numDevices)
      : peerAccessList(numDevices),
        computeStream(0),
        dfExchangeStream(0),
        dfTExchangeStream(0) {}
};

/**
 * @brief Halo exchange parameters for a partition
 *
 */
class HaloExchangeParams {
 public:
  int maxHaloSize;  //! The largest number of halo indices in any df direction
  int nq;           //!< Number of directions (Q) in the distribution functions
  real *srcDfPtr;   //!< The distribution function of this source GPU partition
  int srcQStride;   //!< Number of elements for each q in the source
                    //!< distribution function
  device_vector<int *> srcIdxPtrs;  //!< Pointers to the arrays of halo
                                    //!< exchange indices in the first order q
                                    //!< of the source distribution functions.
  device_vector<real *> dstDfPtrs;  //!< Pointers to distribution functions
                                    //!< on different destination GPUs
  device_vector<int> dstQStrides;   //!< Number of elements for each q in the
                                    //!< destination distribution functions
  device_vector<int *> dstIdxPtrs;  //!< Pointers to the arrays of halo exchange
                                    //!< indices in the first order q of dst df.
  device_vector<int>
      idxLengths;  //!< The sizes of the arrays of halo exchange indices
  HaloExchangeParams(int q)
      : nq(q),
        srcDfPtr(NULL),
        srcIdxPtrs(q),
        dstDfPtrs(q),
        dstQStrides(q),
        dstIdxPtrs(q),
        idxLengths(q),
        srcQStride(0),
        maxHaloSize(0) {}
};

/**
 * @brief Structure containing parameters for the CUDA kernel
 *
 */
class ComputeKernelParams {
 public:
  int nx;  //!< Size of the domain on X-axis
  int ny;  //!< Size of the domain on Y-axis
  int nz;  //!< Size of the domain on Z-axis

  real nu;      //!< Viscosity
  real C;       //!< Smagorinsky constant
  real nuT;     //!< Thermal diffusivity
  real Pr;      //!< Prandtl number of air
  real Pr_t;    //!< Turbulent Prandtl number
  real gBetta;  //!< Gravity times thermal expansion
  real Tref;    //!< Reference temperature for Boussinesq
  real Tinit;   //!< Initial temperature

  DistributionFunction *df;      //!< Velocity distribution functions
  DistributionFunction *df_tmp;  //!< Velocity distribution functions (for swap)
  DistributionFunction *dfT;     //!< Temp. distribution functions
  DistributionFunction *dfT_tmp;  //!< Temp. distribution functions (for swap)

  DistributionFunction *avg; /*!< Contains the macroscopic temperature, velocity
                              * (x,y,z components) integrated in time (so
                              * /nbr_of_time_steps to get average) */

  VoxelArray *voxels;                     //!< The array of voxels
  device_vector<BoundaryCondition> *bcs;  //!< The boundary conditions

  ComputeKernelParams()
      : nx(0),
        ny(0),
        nz(0),
        nu(0),
        C(0),
        nuT(0),
        Pr(0),
        Pr_t(0),
        gBetta(0),
        Tref(0),
        Tinit(0),
        df(nullptr),
        df_tmp(nullptr),
        dfT(nullptr),
        dfT_tmp(nullptr),
        voxels(nullptr),
        bcs(nullptr) {}

  ComputeKernelParams &operator=(const ComputeKernelParams &kp) {
    nx = kp.nx;
    ny = kp.ny;
    nz = kp.nz;
    nu = kp.nu;
    C = kp.C;
    nuT = kp.nuT;
    Pr = kp.Pr;
    Pr_t = kp.Pr_t;
    gBetta = kp.gBetta;
    Tref = kp.Tref;
    Tinit = kp.Tinit;
    return *this;
  }

  ~ComputeKernelParams() {
    delete df;
    delete df_tmp;
    delete dfT;
    delete dfT_tmp;
    delete avg;
    delete voxels;
  }

  void init(const VoxelArray *voxelArray,
            const BoundaryConditionsArray *boundaryConditions) {
    for (int q = 0; q < 4; q++) avg->fill(q, 0);
    avg->upload();

    voxels = new VoxelArray(*voxelArray);
    voxels->upload();
    bcs = new device_vector<BoundaryCondition>(*boundaryConditions);
  }

  void allocate(Partition partition) {
    df->allocate(partition);
    df_tmp->allocate(partition);
    dfT->allocate(partition);
    dfT_tmp->allocate(partition);
    avg->allocate(partition);
  }
};

void buildHaloExchangeParams(HaloExchangeParams *hp, DistributionFunction *df,
                             std::vector<DistributionFunction *> *neighbourDfs,
                             Partition partition);
void runHaloExchangeKernel(HaloExchangeParams *hp, cudaStream_t stream);

/**
 * @brief Class responsible for calling the CUDA kernel
 *
 */
class KernelInterface {
 private:
  // Number of CUDA devices
  int m_numDevices;

  // Cuda kernel parameters
  std::vector<ComputeKernelParams *> m_computeParams;
  std::vector<HaloExchangeParams *> m_dfHaloParams;
  std::vector<HaloExchangeParams *> m_dfTHaloParams;
  std::vector<DeviceParams *> m_deviceParams;

  std::unordered_map<Partition, int> m_partitionDeviceMap;
  std::vector<Partition> m_devicePartitionMap;

  void runComputeKernel(Partition partition, ComputeKernelParams *kp,
                        real *plotGpuPointer,
                        DisplayQuantity::Enum displayQuantity,
                        cudaStream_t computeStream);

  bool enablePeerAccess(int srcDev, int dstDev,
                        std::vector<bool> *peerAccessList);
  void disablePeerAccess(int srcDev, std::vector<bool> *peerAccessList);

 public:
  void runInitKernel(DistributionFunction *df, DistributionFunction *dfT,
                     Partition partition, float rho, float vx, float vy,
                     float vz, float T);
  void uploadBCs(BoundaryConditionsArray *bcs);
  void resetAverages();
  void compute(real *plotGpuPtr, DisplayQuantity::Enum dispQ);
  KernelInterface(const ComputeKernelParams *params,
                  const BoundaryConditionsArray *bcs, const VoxelArray *voxels,
                  const int numDevices);
  ~KernelInterface();
};
