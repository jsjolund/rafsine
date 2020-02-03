from .wrapper.python_lbm import PythonClient
import pandas as pd


class Simulation(PythonClient):
    def __init__(self, lbmFile):
        """Load a simulation from lbm file
        
        Arguments:
            lbmFile {str} -- File system path to the lbm file
        """
        PythonClient.__init__(self, lbmFile)

    def get_average_names(self):
        """List the names of the time averaged measurement areas

        Returns:
            list -- List of name strings
        """
        return super().get_average_names()

    def get_averages(self, avg_type):
        """Get the time averaged values measured during the simulation

        Arguments:
            avg_type {str} -- temperature, velocity or flow

        Returns:
            DataFrame -- A dataframe of measured values for all areas
        """
        return pd.DataFrame(
            data=[[row[0], *(r[avg_type] for r in row[1])]
                  for row in super().get_averages()],
            columns=self.get_average_names())

    def get_boundary_condition_names(self):
        """Get a list of named boundary conditions

        Returns:
            list -- List of name strings
        """
        return super().get_boundary_condition_names()

    def get_boundary_conditions(self, name=None):
        """Get the current boundary conditions

        Keyword Arguments:
            name {str} -- A named boundary condition (default: {None})

        Returns:
            DataFrame -- A dataframe of the boundary conditions by this name or all if None
        """
        bcs = pd.DataFrame(columns=['type', 'temperature',
                                    'velocity', 'normal', 'rel_pos'],
                           data=[(bc.type, bc.temperature,
                                  bc.velocity, bc.normal, bc.rel_pos)
                                 for bc in super().get_boundary_conditions()])
        if name is None:
            return bcs
        else:
            ids = super().get_boundary_condition_ids_from_name(name)
            return bcs.iloc[ids]

    def get_time(self):
        """Get current date and time in the simulation domain

        Returns:
            datetime -- Current date and time
        """
        return super().get_time()

    def get_time_step(self):
        """Get the seconds of simulated time for one discrete time step

        Returns:
            float -- Seconds simulated during one time step
        """
        return super().get_time_step()

    def run(self, seconds):
        """Run the simulation for a number of seconds of simulated time

        Arguments:
            seconds {float} -- Number of seconds to simulate
        """
        super().run(seconds)

    def set_boundary_conditions(self, names, temperatures, vol_flows):
        """Set temperature and volumetric flow for named boundary conditions

        Arguments:
            names {list[str]} -- Name strings
            temperatures {list[float]} -- Temperatures to set
            vol_flows {list[float]} -- Volumetric flows to set
        """
        if len(names) == len(temperatures) == len(vol_flows):
            for i in range(0, len(names)):
                super().set_boundary_condition(names[i], temperatures[i], vol_flows[i])
            super().upload_boundary_conditions()
        else:
            raise RuntimeWarning('List lengths not equal')

    def set_time_averaging_period(self, seconds):
        """Set the time averaging period in seconds

        Arguments:
            seconds {float} -- Number of seconds to average over
        """
        super().set_time_averaging_period(seconds)
