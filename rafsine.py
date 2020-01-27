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
        return pd.DataFrame(data=[(bc.id, bc.type, bc.temperature,
                                   bc.velocity, bc.normal, bc.rel_pos)
                                  for bc in super().get_boundary_conditions()])


def main():
    sim = Simulation('/home/ubuntu/rafsine/problems/jet_chamber/jet_chamber.lbm')

    bc_names = sim.get_boundary_condition_names()
    print(f'Boundary condition names:\n{bc_names}')
    sim.set_time_averaging_period(5.0)

    print(f'Simulation time step: {sim.get_time_step()}')
    print(f'Boundary conditions:\n{sim.get_boundary_conditions()}')

    print(f'Simulation start: {sim.get_time()}')
    sim.run(10.0)
    print(f'Simulation end: {sim.get_time()}')

    print(f'Average temperatures:\n{sim.get_averages("temperature")}')
    print(f'Average velocities:\n{sim.get_averages("velocity")}')
    print(f'Average flows:\n{sim.get_averages("flow")}')

    print('Setting new boundary conditions')
    sim.set_boundary_condition('input', 100, 1.0)

    print(f'Simulation start: {sim.get_time()}')
    sim.run(10.0)
    print(f'Simulation end: {sim.get_time()}')

    print(f'Average temperatures:\n{sim.get_averages("temperature")}')
    print(f'Average velocities:\n{sim.get_averages("velocity")}')
    print(f'Average flows:\n{sim.get_averages("flow")}')

if __name__ == "__main__":
    main()
