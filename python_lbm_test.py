
def main():
    import python_lbm
    sim = python_lbm.Simulation('/home/ubuntu/rafsine/problems/pod2/pod2.lbm')
    # sim.set_time_averaging_period(1.0)

    print(f'Simulation start: {sim.get_time()}')
    sim.run(31.0)
    print(f'Simulation end: {sim.get_time()}')

    print('Averages:')
    averages = sim.get_time_averages()
    for avg in averages:
        print(f'{avg.time}, ', end='')
        for m in avg.measurements:
            print(f'{m.name}, t={m.temperature}, v={m.velocity}, q={m.flow}', end='')
        print()

    print('Boundary conditions')
    bcs = sim.get_boundary_conditions()
    for bc in bcs:
        print(f'id={bc.id}, type={bc.type}, temperature={bc.temperature}, velocity={bc.velocity}, normal={bc.normal}, rel_pos={bc.rel_pos}')


if __name__ == "__main__":
    import os
    import sys
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    main()
