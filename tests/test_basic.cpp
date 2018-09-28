#include <omp.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <gtest/gtest.h>

#include "PartitionTopology.hpp"
#include "DistributionFunctionsGroup.hpp"
#include "LuaContext.hpp"

#define CUDA_RT_CALL(call)                                                                             \
    {                                                                                                  \
        cudaError_t cudaStatus = call;                                                                 \
        if (cudaSuccess != cudaStatus)                                                                 \
            fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);            \
    }

TEST(BasicTopology, Zero)
{
    int nx = 512, ny = 511, nz = 510;
    int divisions = 0;
    Topology topology(nx, ny, nz, divisions);
    Partition *p0 = topology.getPartition(0, 0, 0);
    EXPECT_EQ(p0->getNx(), 512);
    EXPECT_EQ(p0->getNy(), 511);
    EXPECT_EQ(p0->getNz(), 510);
}

TEST(BasicTopology, One)
{
    int nx = 128, ny = 128, nz = 256;
    int divisions = 1;
    Topology topology(nx, ny, nz, divisions);
    Partition *p0 = topology.getPartition(0, 0, 0);
    EXPECT_EQ(p0->getNx(), 128);
    EXPECT_EQ(p0->getNy(), 128);
    EXPECT_EQ(p0->getNz(), 128);
    Partition *p1 = topology.getPartition(0, 0, 1);
    EXPECT_EQ(p1->getNx(), 128);
    EXPECT_EQ(p1->getNy(), 128);
    EXPECT_EQ(p1->getNz(), 128);
}

TEST(BasicTopologyKernel, One)
{
    int nx = 128, ny = 128, nz = 256;
    int numDevices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&numDevices));
    int divisions = 0;
    while (1 << divisions < numDevices)
        divisions++;
    std::cout << "creating " << (1 << divisions) << " partitions" << std::endl;

    // DistributionFunctionsGroup *dfs[numDevices];
    // for (int i = 0; i < numDevices; i++)
    //     dfs[i] = new DistributionFunctionsGroup(19, nx, ny, nz, divisions);

#pragma omp parallel num_threads(numDevices)// shared(dfs)
    {
        int devId = omp_get_thread_num();
        CUDA_RT_CALL(cudaSetDevice(devId));
        CUDA_RT_CALL(cudaFree(0));
        CUDA_RT_CALL(cudaDeviceReset());
        std::cout << devId << std::endl;
    }

    // Partition *p0 = topology.getPartition(0, 0, 0);
    // Partition *p1 = topology.getPartition(0, 0, 1);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}