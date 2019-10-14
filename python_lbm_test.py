import python_lbm
import os
import sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

sim = python_lbm.Simulation('/home/ubuntu/rafsine/problems/pod2/pod2.lbm')
sim.run(1.0)
sim.get_time()

bcs = sim.get_boundary_conditions()
for bc in bcs:
    print(f'id={bc.id}, type={bc.type}, temperature={bc.temperature}, velocity={bc.velocity}, normal={bc.normal}, rel_pos={bc.rel_pos}')
