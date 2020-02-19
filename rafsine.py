#!/usr/bin/python3

from python.simulation import Simulation
import pandas as pd
import os


def main():
    sim = Simulation(f'{os.getcwd()}/problems/jet_chamber/jet_chamber.lbm')

    bc_names = sim.get_boundary_condition_names()
    print(f'Boundary condition names:')
    print(bc_names)

    print(f'Boundary conditions:')
    print(sim.get_boundary_conditions())

    sim.set_time_averaging_period(5.0)
    print(f'Simulation time step: {sim.get_time_step()} seconds')

    print(f'Simulation start: {sim.get_time()}')
    sim.run(30.0)
    print(f'Simulation end: {sim.get_time()}')

    print('Current input boundary conditions')
    bc = sim.get_boundary_conditions(['input', 'output'])
    print(bc)

    print('Setting new input boundary condition')
    sim.set_boundary_conditions(
        ['input', 'output'], [100, float('NaN')], [1.0, 1.0])
    bc = sim.get_boundary_conditions(['input', 'output'])
    print(bc)

    print(f'Simulation start: {sim.get_time()}')
    sim.run(90.0)
    print(f'Simulation end: {sim.get_time()}')

    print('Average temperatures:')
    print(sim.get_averages("temperature"))
    print('Average velocities:')
    print(sim.get_averages("velocity"))
    print('Average flows')
    print(sim.get_averages("flow"))


if __name__ == "__main__":
    main()
