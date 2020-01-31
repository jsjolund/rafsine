#!/usr/bin/python3

import python.python_lbm as lbm
import pandas as pd


class Simulation(lbm.Simulation):
    def __init__(self, lbmFile):
        lbm.Simulation.__init__(self, lbmFile)

    def get_averages(self, avg_type):
        return pd.DataFrame(
            data=[[row[0], *(r[avg_type] for r in row[1])]
                  for row in super().get_averages()],
            columns=self.get_average_names())

    def get_boundary_conditions(self):
        return pd.DataFrame(columns=['type', 'temperature', 
                                    'velocity', 'normal', 'rel_pos'], 
                                    data=[(bc.type, bc.temperature, 
                                    bc.velocity, bc.normal, bc.rel_pos)
                                  for bc in super().get_boundary_conditions()])
    
    def get_boundary_condition(self, name):
        bcs = self.get_boundary_conditions()
        ids = super().get_boundary_condition_ids_from_name(name)
        return bcs.iloc[ids]


def main():
    sim = Simulation('/home/ubuntu/rafsine/problems/jet_chamber/jet_chamber.lbm')

    bc_names = sim.get_boundary_condition_names()
    print(f'Found boundary conditions:\n{bc_names}')
    sim.set_time_averaging_period(5.0)

    print(f'Simulation time step: {sim.get_time_step()} seconds')
    print(f'Boundary conditions:')
    print(sim.get_boundary_conditions())

    print(f'Simulation start: {sim.get_time()}')
    sim.run(10.0)
    print(f'Simulation end: {sim.get_time()}')

    print('Average temperatures:')
    print(sim.get_averages("temperature"))
    print('Average velocities:')
    print(sim.get_averages("velocity"))
    print('Average flows')
    print(sim.get_averages("flow"))

    print('Current input boundary condition')
    bc = sim.get_boundary_condition('input')
    print(bc)

    print('Setting new input boundary condition')
    sim.set_boundary_condition('input', 100, 1.0)
    bc = sim.get_boundary_condition('input')
    print(bc)

    print(f'Simulation start: {sim.get_time()}')
    sim.run(10.0)
    print(f'Simulation end: {sim.get_time()}')

if __name__ == "__main__":
    main()
