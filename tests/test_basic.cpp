#include "PartitionTopology.hpp"

#include <gtest/gtest.h>

TEST(BasicTopology, Zero)
{
    int nx = 512, ny = 511, nz = 510;
    int divisions = 0;
    Topology topology(nx, ny, nz, divisions);
    Partition *p0 = (topology)(0, 0, 0);
    EXPECT_EQ(p0->getNx(), 512);
    EXPECT_EQ(p0->getNy(), 511);
    EXPECT_EQ(p0->getNz(), 510);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}