import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import python_lbm
sim = python_lbm.Simulation('/home/ubuntu/rafsine/problems/pod2/pod2.lbm')
sim.get_time()
sim.run(1.0)
sim.get_time()
bcs = sim.get_boundary_conditions()

