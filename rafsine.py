import sys
import os
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)


def main():
    import python_lbm
    import pandas as pd

    sim = python_lbm.Simulation('/home/ubuntu/rafsine/problems/pod2/pod2.lbm')
    sim.set_time_averaging_period(10.0)

    print(f'Simulation time step: {sim.get_time_step()}')

    print(f'Simulation start: {sim.get_time()}')
    sim.run(120.0)
    print(f'Simulation end: {sim.get_time()}')

    def read_avg(avg_type): return pd.DataFrame(
        data=[[row[0], *(r[avg_type] for r in row[1])]
              for row in sim.get_averages()],
        columns=sim.get_average_names())

    avg_temperature = read_avg('temperature')
    avg_velocity = read_avg('velocity')
    avg_flow = read_avg('flow')

    print('Average temperature:')
    print(avg_temperature)
    print('Average velocity:')
    print(avg_velocity)
    print('Average flow:')
    print(avg_flow)

    bcs = pd.DataFrame(data=[(bc.id, bc.type, bc.temperature,
                        bc.velocity, bc.normal, bc.rel_pos) for bc in sim.get_boundary_conditions()])
    print('Boundary conditions')
    print(bcs)

if __name__ == "__main__":
    main()
