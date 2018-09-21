#include "PartitionTopology.hpp"
#include "LuaContext.hpp"

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

TEST(BasicTopology, One)
{
    int nx = 128, ny = 128, nz = 256;
    int divisions = 1;
    Topology topology(nx, ny, nz, divisions);
    Partition *p0 = (topology)(0, 0, 0);
    EXPECT_EQ(p0->getNx(), 128);
    EXPECT_EQ(p0->getNy(), 128);
    EXPECT_EQ(p0->getNz(), 128);
    Partition *p1 = (topology)(0, 0, 1);
    EXPECT_EQ(p1->getNx(), 128);
    EXPECT_EQ(p1->getNy(), 128);
    EXPECT_EQ(p1->getNz(), 128);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}