from dataclasses import dataclass
from jaxtyping import Float, Array
from typing import Tuple
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter


__all__ = ["Grid", "SourceLocation", "WindField", "AtmosphericState", "SensorsSettings",
        "GaussianPlume", "BackgroundGas", "Sensors"]


def discritize_range(range: Tuple[Float[Array, ""], Float[Array, ""]], dx: Float[Array, ""]):
    """ 
    Discritizing a range.

    Args:
        range (Tuple(Array[float], Array[float])): Range.
        dx (Array[float]): Discritization step.

    Returns:
        (Array[float]): Discritized range.
    """

    return jnp.arange(range[0], range[1] + dx, dx)


@dataclass
class Grid:
    """ 
    Three-dimensional grid specifications. 

    Args:
        x_range (Tuple(Array[float], Array[float])): x dimention in meters.
        y_range (Tuple(Array[float], Array[float])): y dimention in meters.
        z_range (Tuple(Array[float], Array[float])): z dimention in meters.
        dx (Array[float]): x discritization step.
        dy (Array[float]): y discritization step.
        dz (Array[float]): z discritization step.
    """

    x_range: (Array, Array)
    y_range: (Array, Array)
    z_range: (Array, Array)
    dx: Array 
    dy: Array
    dz: Array


    @property
    def x(self):
        """ Discritizes the x range."""
        return discritize_range(self.x_range, self.dx)
    
    @property
    def y(self):
        """ Discritizes the y range."""
        return discritize_range(self.y_range, self.dy)
    
    @property
    def z(self):
        """ Discritizes the z range."""
        return discritize_range(self.z_range, self.dz)



@dataclass
class SourceLocation:
    """ 
    Source location in 3D space specifications.

    Args:
        source_location_x (Array[float]): Source location x in meters.
        source_location_y (Array[float]): Source location y in meters.
        source_location_z (Array[float]): Source location z in meters.
    """

    source_location_x: Array
    source_location_y: Array
    source_location_z: Array



@dataclass
class WindField:
    """ 
    Wind field specifications.

    Args:
        initial_wind_speed (Array[float]): Initial wind speed in m/s.
        initial_wind_direction (Array[float]): Initial wind direction in mathematical degrees.
        end_wind_direction (Array[float]): End wind direction in mathematical degrees.
        Ornstein_Uhlenbeck (bool): True if want incremental changing wind direction instead of OU process-based.
        number_of_time_steps (int): Number of time steps.
        time_step (Array[float]): Time step.
        wind_speed_temporal_std (Array[float]): Wind speed temporal standard deviation.
        wind_direction_temporal_std (Array[float]): Wind direction temporal standard deviation.
        wind_temporal_correlation (Array[float]): Wind temporal correlation.
        wind_speed_seed (int): Wind speed seed.
        wind_direction_seed (int): Wind direction seed.
    """

    initial_wind_speed: Array
    initial_wind_direction: Array
    end_wind_direction: Array
    Ornstein_Uhlenbeck: bool
    number_of_time_steps: Array
    time_step: Array
    wind_speed_temporal_std: Array
    wind_direction_temporal_std: Array
    wind_temporal_correlation: Array
    wind_speed_seed: Array
    wind_direction_seed: Array



@dataclass
class AtmosphericState:
    """ 
    Atmospheric conditions specifications.
    
    Args:
        emission_rate (Array[float]): Emission rate in kg/s.
        source_half_width (Array[float]): Source half width in meters.
        max_abl (Array[float]): Maximum atmospheric boundary layer height in meters.
        background_mean (Array[float]): Mean background concentration in ppm.
        background_std (Array[float]): Standard deviation background concentration in ppm.
        background_seed (int): Seed for the background concentration.
        background_filter (str): Filter to apply to the background concentration (e.g. Gaussian).
        Gaussian_filter_kernel (Array[float]): Controls Gaussian filter smoothness.
        horizontal_opening_angle (Array[float]): Horizontal opening angle.
        vertical_opening_angle (Array[float]): Vertical opening angle.
        a_horizontal (Array[float]): Horizontal parameter.
        a_vertical (Array[float]): Vertical parameter.
        b_horizontal (Array[float]): Horizontal parameter.
        b_vertical (Array[float]): Vertical parameter.
    """
    
    emission_rate: Array
    source_half_width: Array
    max_abl: Array
    background_mean: Array                  
    background_std: Array                   
    background_seed: Array                  
    background_filter: str                  
    Gaussian_filter_kernel: Array           
    horizontal_opening_angle: Array
    vertical_opening_angle: Array
    a_horizontal: Array
    a_vertical: Array
    b_horizontal: Array                 
    b_vertical: Array                 



@dataclass
class SensorsSettings:
    """ 
    Ground sensors' specifications.
    
    Args:
        layout (str): Layout of the sensors. Choose 'grid', 'line', 'random' or 'circle'.
        sensor_number (Array[int]): Number of sensors.
        measurement_error_var (Array[float]): Measurement error variance in ppm.
        sensor_seed (Array[int]): Sensor seed if using random sensor locations.
        measurement_error_seed (Array[int]): Measurement error seed.
        sensor_locations (list[list]): Sensor locations.
    """
    layout: str
    sensor_number: Array
    measurement_error_var: Array
    sensor_seed: Array
    measurement_error_seed: Array
    sensor_locations: list

    @property
    def number_of_different_backgrounds(self):
        if self.layout == "grid":
            if int(self.sensor_number/np.sqrt(self.sensor_number)) == np.sqrt(self.sensor_number) :
                return int(np.sqrt(self.sensor_number))
            else:
                raise ValueError("Invalid number of sensors for a grid. Choose a square number.")
        elif self.layout == "line":
            return int(self.sensor_number)
        elif self.layout == "random":
            return int(self.sensor_number)
        elif self.layout == "circle":
            return int(self.sensor_number)
        else:
            raise ValueError("Invalid layout. Choose 'grid', 'line', 'random' or 'circle'.")


    def grid_of_sensors(p1, p2, p3, p4, num_points=100, formation="grid"):
        """
        Create a grid of equally spaced points.

        Args:
            p1, p2, p3, p4: Tuples of three numbers each, representing the (x,y,z) coordinates in meters of the four points.
            num_points (int): The number of points in each dimension of the grid. Default is 100.
            formation (str): The formation of the grid. Default is "grid". Other option is "line".

        Returns:
            A 2D numpy array of shape (num_points, num_points, 2), representing the (y,z) coordinates of the points in the grid.
        """
        if formation == "grid":
            if int(num_points/np.sqrt(num_points)) == np.sqrt(num_points) :
                num_points = int(np.sqrt(num_points))
            else:
                raise ValueError("Invalid number of sensors for a grid. Choose a square number.")
        # Extract the y and z coordinates of the points
        # x_coords = p1[0]
        x_coords = [p[0] for p in [p1, p2, p3, p4]]
        y_coords = [p[1] for p in [p1, p2, p3, p4]]
        z_coords = [p[2] for p in [p1, p2, p3, p4]]

        # Determine the range of the y and z coordinates
        x_range = (min(x_coords), max(x_coords))
        y_range = (min(y_coords), max(y_coords))
        z_range = (min(z_coords), max(z_coords))

        # Create the grid of points
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.linspace(y_range[0], y_range[1], num_points)
        if formation == "grid":
            z_values = np.linspace(z_range[0], z_range[1], num_points)
        elif formation == "line":
            z_values = 5.0
        else:
            raise ValueError("Invalid formation. Choose 'grid' or 'line'.")
        x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_values, z_values)

        # Combine the y and z coordinates into a single 2D array
        grid = np.stack((x_grid, y_grid, z_grid), axis=-1)

        x_coords = grid[:,:,:,0].flatten()
        y_coords = grid[:,:,:,1].flatten()
        z_coords = grid[:,:,:,2].flatten()

        sensor_locations = [[x_coords[i], y_coords[i], z_coords[i]] for i in range(len(x_coords))]

        return np.unique(sensor_locations, axis=0)
    

    def random_sensor_locations(num_sensors, x_range, y_range, z_range):
        """
        Generate random sensor locations in a specified three dimensional area.

        Args:
            num_sensors (int): The number of sensors to generate.
            x_range (Array[float]): The range of x coordinates (min, max) in meters.
            y_range (Array[float]): The range of y coordinates (min, max) in meters.
            z_range (Array[float]): The range of z coordinates (min, max) in meters.

        Returns:
            A 2D numpy array of shape (num_sensors, 3), representing the (x,y,z) coordinates of the sensors.
        """

        # Generate random x, y, z coordinates
        x_coords = np.random.uniform(x_range[0], x_range[1], num_sensors)
        y_coords = np.random.uniform(y_range[0], y_range[1], num_sensors)
        z_coords = np.random.uniform(z_range[0], z_range[1], num_sensors)

        # Combine the coordinates into a single array
        locations = np.stack((x_coords, y_coords, z_coords), axis=-1)

        return locations
    

    def circle_of_sensors(num_sensors, center, radius):
        """
        Generate sensor locations in a circle.

        Args:
            num_sensors (int): The number of sensors to generate.
            center (Array[float]): The (x,y,z) coordinates in meters of the center of the circle.
            radius (float): The radius of the circle in meters.
        """

        # Angles for each sensor
        angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)

        # Calculate coordinates for sensors
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        z = np.full_like(x, center[2])  # All sensors at the same z-coordinate

        # Combine x, y, z into a 2D array
        circle_sensors = np.vstack((x, y, z)).T

        return circle_sensors
    


    def plot_points_3d(grid, save = False, elev=20, azim=70):
        """
        Plot the points on the grid in 3D space.

        Args:
            grid: A 3D numpy array of shape (num_points, num_points, num_points, 3), representing the (x,y,z) coordinates of the points in the grid.
            elev (float): The elevation viewing angle (in degrees). Default is 30.
            azim (float): The azimuth viewing angle (in degrees). Default is 30.

        Returns:
            (plt.plot): A 3D plot of the points in the grid.
        """

        plt.style.use('default')
        colors = cm.plasma(np.linspace(0.40, 1, len(grid)))

        # Extract the x, y and z coordinates from the grid
        x_coords = [grid[:][i][0] for i in range(len(grid))]
        y_coords = [grid[:][i][1] for i in range(len(grid))]
        z_coords = [grid[:][i][2] for i in range(len(grid))]

        # Create the plot
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(grid)):
            ax.scatter(x_coords[i], y_coords[i], z_coords[i], s=55, c=colors[i], marker='*')
        ax.set_xlabel('$x$: location (m)', fontsize=24, labelpad=15)
        ax.set_ylabel('$y$: location (m)', fontsize=24, labelpad=15)
        ax.set_zlabel('$z$: location (m)', fontsize=24, labelpad=15)
        ax.tick_params(axis='both', labelsize=22)

        # Pivot the plot
        ax.view_init(elev, azim)

        # Flip the y-axis
        ax.invert_yaxis()

        # Manually adjust subplot positions
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        if save == True:
            plt.savefig('sensor_setting_grid.png', format='png', dpi=300, bbox_inches='tight', transparent=False)

        plt.show()


    def plot_sensor_location(locations, save):
        """
        Plot the sensor locations from random_sensor_locations() or circle_of_sensors().
        """
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(locations[:, 0], locations[:, 1], locations[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if save:
            plt.savefig("sensor_locations.png", dpi=300, bbox_inches='tight')
        plt.show()





class GaussianPlume(Grid, SourceLocation, WindField, AtmosphericState, SensorsSettings):
    """
    This class is used to generate the Gaussian plume model for gas dispersion. Both for simulations and the Chilbolton dataset.
    And for grid-based and grid-free source location inversions. 
    
    For the simulated scenarios, this class contains plotting functions for the Gaussian plume model concentrations and wind fields.
    """

    def __init__(self, grid, source_location, wind_field, atmospheric_state, sensors_settings):
        
        self.grid = grid
        self.source_location = source_location
        self.wind_field = wind_field
        self.atmospheric_state = atmospheric_state
        self.sensors_settings = sensors_settings



    def generate_ornstein_u_process(self, parameters, key):
        """
        Generate a time varying Ornstein-Uhlenbeck process of input parameter.
        
        Args:
            parameters (Array[float]): Parameters of the OU process.
            key (Array[int]): PRNG key.
        Returns:
            (Array[float]): An Ornstein-Uhlenbeck process.

        """

        [time_varying_array, sigma, mean, iterations] = parameters
        drift = self.wind_field.wind_temporal_correlation * (mean[iterations] - time_varying_array) * self.wind_field.time_step 
        diffusion = sigma * jnp.sqrt(self.wind_field.time_step) * jax.random.normal(key)
        time_varying_array = time_varying_array + drift + diffusion
        parameters = [time_varying_array, sigma, mean, iterations+1]
        
        return parameters, parameters



    def wind_speed(self):
        """
        Uses a OU process and the WindField class to generate time varying wind speeds.

        Returns:
            (Array[float]): Time varying wind speeds in m/s.

        """

        key = jax.random.PRNGKey(self.wind_field.wind_speed_seed)
        initial_value, mean = self.wind_field.initial_wind_speed, np.repeat(self.wind_field.initial_wind_speed, self.wind_field.number_of_time_steps)
        std = self.wind_field.wind_speed_temporal_std
        _, wind_speed = jax.lax.scan(self.generate_ornstein_u_process, [initial_value, std, mean,0], jax.random.split(key, self.wind_field.number_of_time_steps))
        wind_speed = jnp.where(wind_speed[0] < 1.0, 1.0, wind_speed[0])
        
        return wind_speed



    def wind_direction(self, constant_mean=True, num_periods=1):
        """
        Uses the WindField class to generate time varying wind directions.

        Returns:
            (Array[float]): Time varying wind directions in degrees.
        """

        # Wind directions change incrementally.
        if self.wind_field.Ornstein_Uhlenbeck == False:
            wind_direction = jnp.arange(self.wind_field.initial_wind_direction, self.wind_field.end_wind_direction, (self.wind_field.end_wind_direction - self.wind_field.initial_wind_direction)/self.wind_field.number_of_time_steps)
            assert len(wind_direction) == self.wind_field.number_of_time_steps, "Length of wind_direction must be equal to number_of_time_steps. Choose different number of time steps."
        
        # Wind directions change according to an OU process.
        else:
            key = jax.random.PRNGKey(self.wind_field.wind_direction_seed)
            initial_value, mean = self.wind_field.initial_wind_direction, np.repeat(self.wind_field.initial_wind_direction, self.wind_field.number_of_time_steps)
            std = self.wind_field.wind_direction_temporal_std
            iteration = 0
            if constant_mean == True:
                _, wind_d = jax.lax.scan(self.generate_ornstein_u_process, [initial_value, std, mean, iteration], jax.random.split(key, self.wind_field.number_of_time_steps))
                wind_direction = wind_d[0]
            elif constant_mean == False:
                x = np.linspace(0, num_periods * 2 * np.pi, self.wind_field.number_of_time_steps)
                y = np.sin(x)
                mean = self.wind_field.initial_wind_direction + (self.wind_field.end_wind_direction - self.wind_field.initial_wind_direction) * (y - np.min(y)) / (np.max(y) - np.min(y))
                _, wind_d = jax.lax.scan(self.generate_ornstein_u_process, [initial_value, std, mean, iteration], jax.random.split(key, self.wind_field.number_of_time_steps))
                wind_direction = wind_d[0]
                            
        return wind_direction
    


    def structure_to_vectorise(self, sensor_x, sensor_y, sensor_z = None, number_of_time_steps = None):
        """
        Vectorizes the spatial grid used in grid-based inversion and creates vectors of temporal sensor locations.
        
        Args:
            sensor_x (Array[float]): Sensor x locations in meters.
            sensor_y (Array[float]): Sensor y locations in meters.
            sensor_z (Array[float]): Sensor z locations in meters.
            number_of_time_steps (int): Number of time steps.

        Returns:
            source_coord_grid (Array[float]): Vector of potential source locations in meters.
            temporal_sensor_x (Array[float]): Vector of temporal sensor x locations in meters.
            temporal_sensor_y (Array[float]): Vector of temporal sensor y locations in meters.
            temporal_sensor_z (Array[float]): Vector of temporal sensor z locations in meters.
        """

        # Create a meshgrid of potential source locations from the Grid class.
        if sensor_z is None:
            meshedgrid = jnp.meshgrid(jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx, self.grid.dx) \
                                    , jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy, self.grid.dy))
            sx, sy = meshedgrid[0], meshedgrid[1]
            # Flatten the meshgrid to create a vector of potential source locations.
            source_coord_grid = jnp.array([sx.flatten(), sy.flatten(), np.full(sx.size, self.source_location.source_location_z[0])]).T
        else:
            meshedgrid = jnp.meshgrid(jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx, self.grid.dx) \
                                    , jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy, self.grid.dy) \
                                    , jnp.arange(self.grid.z_range[0], self.grid.z_range[1] + self.grid.dz, self.grid.dz))
            sx, sy, sz = meshedgrid[0], meshedgrid[1], meshedgrid[2]
            # Flatten the meshgrid to create a vector of potential source locations.
            source_coord_grid = jnp.array([sx.flatten(), sy.flatten(), sz.flatten()]).T

        # Repeat the sensor locations for each time step.
        if number_of_time_steps is None:
            temporal_sensor_x = jnp.repeat(sensor_x, self.wind_field.number_of_time_steps)
            temporal_sensor_y = jnp.repeat(sensor_y, self.wind_field.number_of_time_steps)
            if sensor_z is None:
                temporal_sensor_z = jnp.repeat(self.source_location.source_location_z[0], self.wind_field.number_of_time_steps)
            else:
                temporal_sensor_z = jnp.repeat(sensor_z, self.wind_field.number_of_time_steps)
        else:
            temporal_sensor_x = jnp.repeat(sensor_x, number_of_time_steps)
            temporal_sensor_y = jnp.repeat(sensor_y, number_of_time_steps)
            if sensor_z is None:
                temporal_sensor_z = jnp.repeat(self.source_location.source_location_z[0], number_of_time_steps)
            else:
                temporal_sensor_z = jnp.repeat(sensor_z, number_of_time_steps)

        return source_coord_grid, temporal_sensor_x, temporal_sensor_y, temporal_sensor_z



    def downwind_distance(self, s, coupling_x, coupling_y, wd):
        """
        Compute the downwind distances. This is the distances between the source and point
        at which we want to compute the Gaussian plume model concentration. When doing inversion,
        this corresponds to the distances between the source and the sensors.

        Args:
            s (Array[float]): Source location in meters.
            x (Array[float]): Sensor x coordinate in meters.
            y (Array[float]): Sensor y coordinate in meters.
        
        Returns:
            (Array[float]): Downwind distances in meters.
        """

        # Convert wind direction to radians.
        wd = jnp.deg2rad(wd % 360)

        return jnp.cos(wd) * (coupling_x - s[0,:].reshape(-1,1).T) + jnp.sin(wd) * (coupling_y - s[1,:].reshape(-1,1).T)



    def horizontal_stddev(self, downwind, a_horizontal, b_horizontal, tanH, estimated = False, scheme="Draxler", stability_class = "B"):
        """
        Compute the horizontal standard deviation of the Gaussian plume model. i.e horizontal wind sigma.

        Args:
            downwind (Array[float]): Downwind distances in meters.
            a_horizontal (Array[float]): Horizontal dispersion parameter.
            b_horizontal (Array[float]): Horizontal dispersion parameter.
            tanH (Array[float]): Tangent of the rolling standard deviation of the horizontal wind direction.
            estimated (bool): True if the dispersion parameters are estimated.
            scheme (str): Dispersion parameter scheme.
            stability_class (str): Pasquill's Atmospheric stability class.

        Returns:
            (Array[float]): Horizontal standard deviation i.e. horizontal wind sigma.
        """
        
        # Ensure downwind distances are positive. Setting negative to zero ensures no upwind Gaussian plume model contributions.
        posi_dw = jnp.where(downwind < 0.0, 1.0, downwind)

        # Using Atmospheric stability class-based dispersion parameters schemes.
        if estimated == False:
            if scheme == "Draxler":
                horizontal_std = ( self.atmospheric_state.a_horizontal * jnp.power((jnp.tan(jnp.deg2rad(self.atmospheric_state.horizontal_opening_angle)) * posi_dw), self.atmospheric_state.b_horizontal) ) + self.atmospheric_state.source_half_width
            elif scheme == "SMITH":
                if stability_class == "B":
                    horizontal_std = 0.4 * jnp.power(posi_dw, 0.91)
                elif stability_class == "C":
                    horizontal_std = 0.36 * jnp.power(posi_dw, 0.86)
                elif stability_class == "D":
                    horizontal_std = 0.32 * jnp.power(posi_dw, 0.78)
            elif scheme == "Briggs":
                if stability_class == "A":
                    horizontal_std = 0.22 * posi_dw * (1 + 0.0001 * posi_dw)**(-0.5)
                elif stability_class == "B":
                    horizontal_std = 0.16 * posi_dw * (1 + 0.0001 * posi_dw)**(-0.5)
                elif stability_class == "C":
                    horizontal_std = (0.11 * posi_dw) * ((1 + 0.0002 * posi_dw)**(-0.5))
                elif stability_class == "D":
                    horizontal_std = (0.08 * posi_dw) * ((1 + 0.0015 * posi_dw)**(-0.5))
                elif stability_class == "E":
                    horizontal_std = (0.06 * posi_dw) * ((1 + 0.0003 * posi_dw)**(-1))
                elif stability_class == "F":
                    horizontal_std = (0.04 * posi_dw) * ((1 + 0.0003 * posi_dw)**(-1))

        # Using estimated dispersion parameters schemes.
        elif estimated == True:
            if scheme == "Draxler":
                horizontal_std = (a_horizontal * jnp.power((tanH*posi_dw), b_horizontal)) + self.atmospheric_state.source_half_width
            elif scheme == "SMITH":
                horizontal_std = a_horizontal * jnp.power(posi_dw, b_horizontal)
        return horizontal_std



    def vertical_stddev(self, downwind, a_vertical, b_vertical, tanV, estimated = False, scheme="Draxler", stability_class = "B"):
        """
        Compute the vertical standard deviation of the Gaussian plume model. i.e vertical wind sigma.

        Args:
            downwind (Array[float]): Downwind distances in meters.
            a_vertical (Array[float]): Vertical dispersion parameter.
            b_vertical (Array[float]): Vertical dispersion parameter.
            tanV (Array[float]): Tangent of the rolling standard deviation of the vertical wind direction.
            estimated (bool): True if the dispersion parameters are estimated.
            scheme (str): Dispersion parameter scheme.
            stability_class (str): Pasquill's Atmospheric stability class.

        Returns:
            (Array[float]): Vertical standard deviation i.e. vertical wind sigma.
        """
        
        # Ensure downwind distances are positive. Setting negative to zero ensures no upwind Gaussian plume model contributions.
        posi_dw = jnp.where(downwind < 0.0, 1.0, downwind)
        
        # Using Atmospheric stability class-based dispersion parameters schemes. 
        if estimated == False:
            if scheme == "Draxler":
                vertical_std = self.atmospheric_state.a_vertical * jnp.power((jnp.tan(jnp.deg2rad(self.atmospheric_state.vertical_opening_angle)) * posi_dw), self.atmospheric_state.b_vertical)
            elif scheme == "SMITH":
                if stability_class == "B":
                    vertical_std = 0.41 * jnp.power(posi_dw, 0.91)
                elif stability_class == "C":
                    vertical_std = 0.33 * jnp.power(posi_dw, 0.86)
                elif stability_class == "D":
                    vertical_std = 0.22 * jnp.power(posi_dw, 0.78)
            elif scheme == "Briggs":
                if stability_class == "A":
                    vertical_std = 0.2 * posi_dw
                elif stability_class == "B":
                    vertical_std = 0.12 * posi_dw
                elif stability_class == "C":
                    vertical_std = (0.08 * posi_dw) * ((1 + 0.0002 * posi_dw)**(-0.5))
                elif stability_class == "D":
                    vertical_std = (0.06 * posi_dw) * ((1 + 0.0015 * posi_dw)**(-0.5))
                elif stability_class == "E":
                    vertical_std = (0.03 * posi_dw) * ((1 + 0.0003 * posi_dw)**(-1))
                elif stability_class == "F":
                    vertical_std = (0.016 * posi_dw) * ((1 + 0.0003 * posi_dw)**(-1))
        
        # Using estimated dispersion parameters schemes.
        elif estimated == True:
            if scheme == "Draxler":
                vertical_std = a_vertical * jnp.power((tanV*posi_dw), b_vertical)
            elif scheme == "SMITH":
                vertical_std = a_vertical * jnp.power(posi_dw, b_vertical)     

        return vertical_std


    def horizontal_offset(self, s, x, y, wd):
        """
        Compute the horizontal offset of the Gaussian plume model at a single point (x,y,z).

        Args:
            s (Array[float]): Source location (x,y,z) in meters.
            x (Array[float]): x coordinate of the point in meters.
            y (Array[float]): y coordinate of the point in meters.
            wd (Array[float]): Wind direction in degrees.

        Returns:
            (Array[float]): Horizontal offset of the Gaussian plume model at a single point (x,y).
        """

        wd = jnp.deg2rad(wd % 360)
        source_x, source_y = s[0,:].reshape(-1,1).T, s[1,:].reshape(-1,1).T

        return  -jnp.sin(wd) * (x - source_x) + jnp.cos(wd) * (y - source_y)



    def vertical_offset(self, z, s):
        """
        Compute the vertical offset of the Gaussian plume model at a single point (x,y,z).

        Args:
            s (Array[float]): Source location (x,y,z) in meters.
            z (Array[float]): z coordinate of the point in meters.

        Returns:
            (Array[float]): Vertical offset of the Gaussian plume model at a single point (x,y,z).
        """

        source_z = s[2,:].reshape(-1,1).T

        return z - source_z



    def methane_kg_m3_to_ppm(self, kg):
        """
        Methane conversion from kg/m^3 to ppm at 15°C and 1 atm.

        Args:
            kg (Array[float]): Gaussian plume model's concentration in kg/m^3.
            0.671: density of methane in kg/m^3 at 15°C and 1 atm.
            
        Returns:
            (Array[float]): Methane concentration in ppm.
        """
        
        return kg * 1e6/0.671



    def gaussian_plume_for_plot(self, source_nbr):
        """
        Compute the Gaussian plume model gas concentration for a single source over
        the grid from Grid class using atmospheric conditions from AtmosphericState class.

        Args:
            source_nbr (int): The source number. 
        
        Returns:
            (Array[float]): Gas concentration in kg/m^3.
        """
        # Source number (source_nbr) location.
        source = jnp.array([[self.source_location.source_location_x[source_nbr]], [self.source_location.source_location_y[source_nbr]]])

        # Creating a grid to compute the Gaussian plume model concentration and plot it.
        meshedgrid = jnp.meshgrid(jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + self.grid.dx, self.grid.dx) \
                    , jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + self.grid.dy, self.grid.dy))
        all_x, all_y = meshedgrid[0].flatten(), meshedgrid[1].flatten()

        # Downwind distance, horizontal and vertical offsets.
        delta_R = self.downwind_distance(source, all_x, all_y, self.wind_field.initial_wind_direction)
        delta_horizontal = self.horizontal_offset(source, all_x, all_y, self.wind_field.initial_wind_direction)
        ground_level = 0.0
        delta_vertical = ground_level - self.source_location.source_location_z[source_nbr]
        # Horizontal and vertical standard deviations. i.e. horizontal and vertical wind sigmas.
        sigma_vertical = self.vertical_stddev(downwind = delta_R, a_vertical = None, b_vertical = None, tanV = None, estimated = False, scheme="Draxler")
        sigma_horizontal = self.horizontal_stddev(downwind = delta_R, a_horizontal = None, b_horizontal = None, tanH = None, estimated = False, scheme="Draxler")
        # Wind speed, emission rate, atmospheric boundary layer height, source height.
        ws = self.wind_field.initial_wind_speed
        er = self.atmospheric_state.emission_rate[source_nbr] 
        max_abl = self.atmospheric_state.max_abl
        height = self.source_location.source_location_z[source_nbr]
        twosigmavsqrd = 2 * (sigma_vertical ** 2)

        # Gaussian plume model gas concentration over the grid.
        concentration = (1/(2 * jnp.pi * ws * sigma_horizontal * sigma_vertical)) * \
                jnp.exp(-(delta_horizontal **2)/(2*(sigma_horizontal **2))) * \
                (jnp.exp(-((delta_vertical )**2)/(twosigmavsqrd)) + \
                jnp.exp(-((delta_vertical  - 2*(max_abl - height))**2)/(twosigmavsqrd)) + \
                jnp.exp(-((2 * height + delta_vertical  )**2)/(twosigmavsqrd)) + \
                jnp.exp(-((2 * max_abl + delta_vertical )**2)/(twosigmavsqrd))) * er
        # Reshaping the concentration.
        concentration = concentration.flatten().squeeze()
        # Setting downwind distances of zero to zero concentration.
        concentration = jnp.where(jnp.isnan(concentration), 0.0, concentration)
        # Setting negative downwind distances to zero concentration.
        concentration = jnp.where(delta_R < 0.0, 0.0, concentration)
        
        return self.methane_kg_m3_to_ppm(concentration)


    def gaussian_plume_plot(self, log_scale = False, save = False, format = "pdf"):
        """
        Plot the Gaussian plume model gas concentration over the grid from Grid class using atmospheric conditions from AtmosphericState class.

        Args:
            log_scale (bool): True if you want to plot the log of the Gaussian plume model gas concentration.
            save (bool): True if you want to save the plot.
            format (str): Format to save the plot in.

        Returns:
            (plt.Plot): Gaussian plume model gas concentration plot.
        """

        if log_scale == False:

            # Compute the Gaussian plume model gas concentration for all sources.
            concentrations = 0
            for source_nbr in range(len(self.source_location.source_location_x)):
                concentrations += self.gaussian_plume_for_plot(source_nbr)
            df = pd.DataFrame( (concentrations + 1e-8).reshape(len(self.grid.y), len(self.grid.x)))

            # Plotting the Gaussian plume model gas concentration.
            plt.figure(figsize=(8, 5))

            # Setting axis labels and ticks.
            if (len(self.grid.x) > 10) and (len(self.grid.y) > 10):
                ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
                ax.set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
                ax.set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
                ax.set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + 3*self.grid.dy, self.grid.y_range[1]/10)], rotation=0)
                ax.set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + 3*self.grid.dx, self.grid.x_range[1]/10)], rotation=0)
            else:
                ax = sns.heatmap(df, cmap="jet")
                ax.set_yticklabels(self.grid.y)
                ax.set_xticklabels(self.grid.x)
            ax.invert_yaxis()

            # Plotting the source locations.
            for source_nbr in range(len(self.source_location.source_location_x)):
                ax.scatter(float(self.source_location.source_location_x[source_nbr]/self.grid.dx), float(self.source_location.source_location_y[source_nbr]/self.grid.dy), marker='.', s=100, color='orange')
            
            # Setting title and labels.
            colorbar = ax.collections[0].colorbar
            colorbar.set_label('Parts per million (PPM)')
            plt.title("Initial Gaussian Plume")
            plt.xlabel("$x$: location (m)")
            plt.ylabel("$y$: location (m)")
            if save == True:
                plt.savefig("Initial Gaussian Plume." + format, dpi=300, bbox_inches="tight")

        elif log_scale == True:

            # Compute the log of the Gaussian plume model gas concentration for all sources.
            concentrations = 0
            for source_nbr in range(len(self.source_location.source_location_x)):
                concentrations += jnp.log(self.gaussian_plume_for_plot(source_nbr) + 1e-8)
            df = pd.DataFrame(concentrations.reshape(len(self.grid.y), len(self.grid.x)))

            # Plotting the log of the Gaussian plume model gas concentration.
            plt.figure(figsize=(8, 5))

            # Setting axis labels and ticks.
            if (len(self.grid.x) > 10) and (len(self.grid.y) > 10):
                ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
                ax.set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
                ax.set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
                ax.set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + 3*self.grid.dy, self.grid.y_range[1]/10)], rotation=0)
                ax.set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + 3*self.grid.dx, self.grid.x_range[1]/10)], rotation=0)
            else:
                ax = sns.heatmap(df, cmap="jet")
                ax.set_yticklabels(self.grid.y)
                ax.set_xticklabels(self.grid.x)
            ax.invert_yaxis()

            # Plotting the source locations.
            for source_nbr in range(len(self.source_location.source_location_x)):
                ax.scatter(float(self.source_location.source_location_x[source_nbr]/self.grid.dx), float(self.source_location.source_location_y[source_nbr]/self.grid.dy), marker='.', s=100, color='orange')
            
            # Setting title and labels.
            colorbar = ax.collections[0].colorbar
            colorbar.set_label('Log parts per million (PPM)')
            plt.title("Log Initial Gaussian Plume")
            plt.xlabel("$x$: location (m)")
            plt.ylabel("$y$: location (m)")
            if save == True:
                plt.savefig("Log initial Gaussian Plume." + format, dpi=300, bbox_inches="tight")
        
        return plt.show()
    


    def fixed_objects_of_grided_coupling_matrix(self, wind_speed = None, wind_direction = None, number_of_time_steps = None, constant_mean = True):
        """ 
        Returns the fixed objects for the temporal coupling matrix for grid-based inversion using the Gaussian plume model.
        Avoids having to recompute them. These are not dependent on parameters being estimated.

        Args:   
            wind_speed (Array[float]): Temporal wind speeds in m/s.
            wind_direction (Array[float]): Temporal wind directions in degrees.
            number_of_time_steps (int): Number of time steps.
        
        Returns:
            windspeeds (Array[float]): Wind speeds in m/s.
            delta_R (Array[float]): Downwind distances in meters.
            delta_horizontal (Array[float]): Horizontal offsets in meters.
            delta_vertical (Array[float]): Vertical offsets in meters.
            max_abl (Array[float]): Atmospheric boundary layer height in meters.
            height (Array[float]): Source height in meters.
        """
        
        # Extracting sensors x, y, z locations.
        sensor_x = jnp.array([i[0] for i in self.sensors_settings.sensor_locations])
        sensor_y = jnp.array([i[1] for i in self.sensors_settings.sensor_locations])
        sensor_z = jnp.array([i[2] for i in self.sensors_settings.sensor_locations])

        # Setting up wind field.
        if wind_speed is None:
            windspeeds = jnp.tile(self.wind_speed(), len(self.sensors_settings.sensor_locations)).reshape(-1,1)
        else:
            windspeeds = wind_speed
        if wind_direction is None:
            winddirection = jnp.tile(self.wind_direction(constant_mean), len(self.sensors_settings.sensor_locations)).reshape(-1,1)
        else:
            winddirection = wind_direction
        
        # Setting source locations to cell centers.
        s = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[0].T
        s = s + self.grid.dx/2

        # Temporal sensor locations.
        if number_of_time_steps is None:
            temporal_sensor_x = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[1].reshape(-1,1)
            temporal_sensor_y = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[2].reshape(-1,1)
            temporal_sensor_z = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[3].reshape(-1,1)
        else:
            temporal_sensor_x = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z, number_of_time_steps)[1].reshape(-1,1)
            temporal_sensor_y = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z, number_of_time_steps)[2].reshape(-1,1)
            temporal_sensor_z = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z, number_of_time_steps)[3].reshape(-1,1)

        # Downwind distances, horizontal and vertical offsets.
        delta_R = self.downwind_distance(s, temporal_sensor_x, temporal_sensor_y, winddirection)
        delta_horizontal = self.horizontal_offset(s, temporal_sensor_x, temporal_sensor_y, winddirection)
        delta_vertical = self.vertical_offset(temporal_sensor_z, s)

        # Atmospheric boundary layer height, source height.
        max_abl = self.atmospheric_state.max_abl
        height = self.source_location.source_location_z

        return windspeeds, delta_R, delta_horizontal, delta_vertical, max_abl, height



    def temporal_grided_coupling_matrix(self, fixed, a_horizontal = None, a_vertical = None, b_horizontal = None, b_vertical = None, simulation = True, scheme = "Draxler", stability_class = "B"):
        """
        Computes the temporal coupling matrix for grid-based inversion using the Gaussian plume model.

        Args:
            fixed : Output from fixed_objects_of_grided_coupling_matrix().
            a_horizontal (Array[float]): Horizontal dispersion parameter.
            a_vertical (Array[float]): Vertical dispersion parameter.
            b_horizontal (Array[float]): Horizontal dispersion parameter.
            b_vertical (Array[float]): Vertical dispersion parameter.
            simulation (bool): True if using the Gaussian plume model for simulation.
            scheme (str): Dispersion parameter scheme.
            stability_class (str): Pasquill's Atmospheric stability class.
        
        Returns:
            A (Array[float]): Temporal coupling matrix for grid-based inversion using the Gaussian plume model.
        """

        if a_horizontal is None:
            a_horizontal = self.atmospheric_state.a_horizontal
        if a_vertical is None:
            a_vertical = self.atmospheric_state.a_vertical
        if b_horizontal is None:
            b_horizontal = self.atmospheric_state.b_horizontal
        if b_vertical is None:
            b_vertical = self.atmospheric_state.b_vertical
        if simulation == True:
            tan_gamma_horizontal, tan_gamma_vertical = jnp.tan(jnp.deg2rad(self.atmospheric_state.horizontal_opening_angle)), jnp.tan(jnp.deg2rad(self.atmospheric_state.vertical_opening_angle))
        elif simulation == False:
            tan_gamma_horizontal, tan_gamma_vertical = fixed[7], fixed[8]

        # Extracting fixed objects.
        windspeeds = fixed[0]
        delta_R, delta_horizontal, delta_vertical = fixed[1], fixed[2], fixed[3]
        max_abl, height = fixed[4], fixed[5]

        # Horizontal and vertical standard deviations.
        sigma_vertical = self.vertical_stddev(delta_R, a_vertical, b_vertical, tan_gamma_vertical, scheme, stability_class)
        sigma_horizontal = self.horizontal_stddev(delta_R, a_horizontal, b_horizontal, tan_gamma_horizontal, scheme, stability_class)
        twosigmavsqrd = 2.0 * (sigma_vertical ** 2)

        # Gaussian plume model gas concentration.
        coupling = ( 1.0/ (2.0 * jnp.pi * windspeeds * sigma_horizontal * sigma_vertical ) ) * \
            jnp.exp( -(delta_horizontal **2 )/(2.0*(sigma_horizontal **2)) ) * \
            (   jnp.exp( -((delta_vertical )**2)/(twosigmavsqrd) ) + \
                jnp.exp( -((delta_vertical  - 2.0*(max_abl - height))**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * height + delta_vertical  )**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * max_abl + delta_vertical )**2)/(twosigmavsqrd)) )

        # Setting negative downwind distances to zero concentration.
        coupling = jnp.where(delta_R <= 0.0, 0.0, coupling)

        # Convert gas concentration from kg/m^3 to ppm.
        A = self.methane_kg_m3_to_ppm(coupling)
        
        return A



    def fixed_objects_of_gridfree_coupling_matrix(self, wind_speed = None, wind_direction = None, number_of_time_steps = None, constant_mean = True):
        """ 
        Returns the fixed objects for the temporal coupling matrix for grid-free inversion using the Gaussian plume model.
        Avoids having to recompute them. These are not dependent on parameters being estimated.

        Args:
            wind_speed (Array[float]): Temporal wind speeds in m/s.
            wind_direction (Array[float]): Temporal wind directions in degrees.
            number_of_time_steps (int): Number of time steps.
        """

        # Extracting sensors x, y, z locations.
        sensor_x = jnp.array([i[0] for i in self.sensors_settings.sensor_locations])
        sensor_y = jnp.array([i[1] for i in self.sensors_settings.sensor_locations])
        sensor_z = jnp.array([i[2] for i in self.sensors_settings.sensor_locations])

        # Setting up wind field.
        if wind_speed is None:
            windspeeds = jnp.tile(self.wind_speed(), len(self.sensors_settings.sensor_locations)).reshape(-1,1)
        else:
            windspeeds = wind_speed
        if wind_direction is None:
            winddirection = jnp.tile(self.wind_direction(constant_mean), len(self.sensors_settings.sensor_locations)).reshape(-1,1)
        else:
            if wind_direction.shape[0] == len(self.sensors_settings.sensor_locations)*self.wind_field.number_of_time_steps:
                winddirection = wind_direction
            else:
                winddirection = jnp.tile(wind_direction, len(self.sensors_settings.sensor_locations)).reshape(-1,1)

        # Temporal sensor locations.
        if number_of_time_steps is None:
            temporal_sensor_x = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[1].reshape(-1,1)
            temporal_sensor_y = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[2].reshape(-1,1)
            temporal_sensor_z = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z)[3].reshape(-1,1)
        else:
            temporal_sensor_x = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z, number_of_time_steps)[1].reshape(-1,1)
            temporal_sensor_y = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z, number_of_time_steps)[2].reshape(-1,1)
            temporal_sensor_z = self.structure_to_vectorise(sensor_x, sensor_y, sensor_z, number_of_time_steps)[3].reshape(-1,1)

        # Atmospheric boundary layer height, source height.
        max_abl = self.atmospheric_state.max_abl
        # height = self.source_location.source_location_z

        # return windspeeds, winddirection, temporal_sensor_x, temporal_sensor_y, max_abl, height, temporal_sensor_z
        return windspeeds, winddirection, temporal_sensor_x, temporal_sensor_y, max_abl, temporal_sensor_z



    def temporal_gridfree_coupling_matrix(self, fixed, x_coord = None, y_coord = None, z_coord = None, a_horizontal = None, a_vertical = None, b_horizontal = None, b_vertical = None, simulation = True, estimated = False, scheme = "Draxler", stability_class = "B"):
        """
        Computes the temporal coupling matrix for grid-free inversion using the Gaussian plume model.

        Args:
            fixed : Output from fixed_objects_of_gridfree_coupling_matrix().
            x_coord (Array[float]): Source x locations in meters.
            y_coord (Array[float]): Source y locations in meters.
            z_coord (Array[float]): Source z locations in meters.
            a_horizontal (Array[float]): Horizontal dispersion parameter.
            a_vertical (Array[float]): Vertical dispersion parameter.
            b_horizontal (Array[float]): Horizontal dispersion parameter.
            b_vertical (Array[float]): Vertical dispersion parameter.
            simulation (bool): True if using the Gaussian plume model for simulation.
            estimated (bool): True if the dispersion parameters are estimated.
            scheme (str): Dispersion parameter scheme.
            stability_class (str): Pasquill's Atmospheric stability class.

        Returns:
            A (Array[float]): Temporal coupling matrix for grid-free inversion using the Gaussian plume model.
        """

        if x_coord is None:
            x_coord = self.source_location.source_location_x
        if y_coord is None:
            y_coord = self.source_location.source_location_y
        if z_coord is None:
            z_coord = self.source_location.source_location_z
        if a_horizontal is None:
            a_horizontal = self.atmospheric_state.a_horizontal
        if a_vertical is None:
            a_vertical = self.atmospheric_state.a_vertical
        if b_horizontal is None:
            b_horizontal = self.atmospheric_state.b_horizontal
        if b_vertical is None:
            b_vertical = self.atmospheric_state.b_vertical
        if simulation == True:
            tan_gamma_horizontal, tan_gamma_vertical = jnp.tan(jnp.deg2rad(self.atmospheric_state.horizontal_opening_angle)), jnp.tan(jnp.deg2rad(self.atmospheric_state.vertical_opening_angle))
        elif simulation == False:
            tan_gamma_horizontal, tan_gamma_vertical = fixed[7], fixed[8]
        
        # Extracting fixed objects.
        windspeeds, winddirection = fixed[0], fixed[1]
        temporal_sensor_x, temporal_sensor_y = fixed[2], fixed[3]
        # max_abl, height = fixed[4], fixed[5]
        max_abl = fixed[4]
        height = z_coord[0]
        temporal_sensor_z = fixed[5]

        # Source locations.
        if z_coord is None:
            s = jnp.array([x_coord, y_coord, jnp.full(len(x_coord), self.source_location.source_location_z)]).reshape(3,len(x_coord))
        else:
            s = jnp.array([x_coord, y_coord, z_coord]).reshape(3,len(x_coord))

        # Downwind distances, horizontal and vertical offsets.
        delta_R = self.downwind_distance(s, temporal_sensor_x, temporal_sensor_y, winddirection)
        delta_horizontal = self.horizontal_offset(s, temporal_sensor_x, temporal_sensor_y, winddirection)
        delta_vertical = self.vertical_offset(temporal_sensor_z, s)

        # Horizontal and vertical standard deviations.
        sigma_vertical = self.vertical_stddev(delta_R, a_vertical, b_vertical, tan_gamma_vertical, estimated, scheme, stability_class)
        sigma_horizontal = self.horizontal_stddev(delta_R, a_horizontal, b_horizontal, tan_gamma_horizontal, estimated, scheme, stability_class)
        twosigmavsqrd = 2.0 * (sigma_vertical ** 2)

        # Gaussian plume model gas concentration.
        coupling = ( 1.0/ (2.0 * jnp.pi * windspeeds * sigma_horizontal * sigma_vertical ) ) * \
            jnp.exp( -(delta_horizontal **2 )/(2.0*(sigma_horizontal **2)) ) * \
            (   jnp.exp( -((delta_vertical )**2)/(twosigmavsqrd) ) + \
                jnp.exp( -((delta_vertical  - 2.0*(max_abl - height))**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * height + delta_vertical  )**2)/(twosigmavsqrd)) + \
                jnp.exp( -((2 * max_abl + delta_vertical )**2)/(twosigmavsqrd)) )

        # Setting negative downwind distances to zero concentration.
        coupling = jnp.where(delta_R <= 0.0, 0.0, coupling)

        # Convert gas concentration from kg/m^3 to ppm.
        A = self.methane_kg_m3_to_ppm(coupling)

        return A



    def fixed_objects_of_gridfree_chilbolton_coupling_matrix(self, simulation = False, wind_speed = None, wind_direction = None, tangamma_ts = None, number_of_time_steps = None, source3and4 = False, release_17 = False):
        """ 
        Returns the fixed objects for the temporal coupling matrix for grid-free inversion using the Gaussian plume model for the Chilbolton dataset.
        The Chilbolton data is recorded using beam sensors. Each beam is converted into 40cm spaced concentration measurement points along the beam
        path which averages to the path-averaged recorded concentration. Each beam concentration is replaced by a different number of point measurement.
        Therefore, each beam path's fixed objects have different dimensions and need to be computed separatly. This function avoids having to recompute
        the fixed objects. These are not dependent on parameters being estimated.
        
        Args:
            simulation (bool): True if using the Gaussian plume model for simulation of sources using the Chilbolton sensor set-up.
            wind_speed (Array[float]): Temporal wind speeds in m/s.
            wind_direction (Array[float]): Temporal wind directions in degrees.
            number_of_time_steps (int): Number of time steps.
            source3and4 (bool): True if needing fixed objects for source 3 and 4.
            release_17 (bool): True if release 17 data is used. One of the sensors is missing in release 17 data.
        
        Returns:
            for reflector k = 1, 2, ..., 7:
                windspeeds_refk (Array[float]): Temporal wind speeds in m/s.
                winddirection_refk (Array[float]): Temporal wind directions in degrees.
                temporal_sensor_x_refk (Array[float]): Temporal measurement point x locations in meters.
                temporal_sensor_y_refk (Array[float]): Temporal measurement point y locations in meters.
                temporal_sensor_z_refk (Array[float]): Temporal measurement point z locations in meters.
                horizontal_temporal_tangamma_ts_refk (Array[float]): Temporal horizontal tangent of the rolling standard deviation of the horizontal wind direction.
                vertical_temporal_tangamma_ts_refk (Array[float]): Temporal vertical tangent of the rolling standard deviation of the vertical wind direction.
            max_abl (Array[float]): Atmospheric boundary layer height in meters.
            height (Array[float]): Source height in meters.
        """

        # Extracting sensors x, y, z locations.

        # Reflector 1.
        sensor_x_ref1 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[0:18*5]])
        sensor_y_ref1 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[0:18*5]])
        sensor_z_ref1 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[0:18*5]])

        # Reflector 2.
        sensor_x_ref2 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[18*5:51*5]])
        sensor_y_ref2 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[18*5:51*5]])
        sensor_z_ref2 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[18*5:51*5]])

        # Reflector 3.
        sensor_x_ref3 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[51*5:73*5]])
        sensor_y_ref3 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[51*5:73*5]])
        sensor_z_ref3 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[51*5:73*5]])

        # Reflector 4.
        sensor_x_ref4 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[73*5:122*5]])
        sensor_y_ref4 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[73*5:122*5]])
        sensor_z_ref4 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[73*5:122*5]])

        # Reflector 5.
        sensor_x_ref5 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[122*5:164*5]])
        sensor_y_ref5 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[122*5:164*5]])
        sensor_z_ref5 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[122*5:164*5]])

        # Reflector 6.
        sensor_x_ref6 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[164*5:193*5]])
        sensor_y_ref6 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[164*5:193*5]])
        sensor_z_ref6 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[164*5:193*5]])

        # Reflector 7.
        sensor_x_ref7 = jnp.array([i[0] for i in self.sensors_settings.sensor_locations[193*5:210*5]])
        sensor_y_ref7 = jnp.array([i[1] for i in self.sensors_settings.sensor_locations[193*5:210*5]])
        sensor_z_ref7 = jnp.array([i[2] for i in self.sensors_settings.sensor_locations[193*5:210*5]])

        # Setting up wind field.

        # If simulation of the Chilbolton set up then use simulated wind field.
        if simulation == True:
            wind_speed = self.wind_speed()
            wind_direction = self.wind_direction()
            horizontal_tangamma = jnp.tan(jnp.deg2rad(self.atmospheric_state.horizontal_opening_angle))
            vertical_tangamma = jnp.tan(jnp.deg2rad(self.atmospheric_state.vertical_opening_angle))
        elif simulation == False:
            vertical_tangamma = tangamma_ts['Average Tan_gamma Vertical'].values
            horizontal_tangamma = tangamma_ts['Average Tan_gamma Horizontal'].values
        
        # Reflector 1.
        windspeeds_ref1 = jnp.tile(wind_speed, sensor_x_ref1.shape[0]).reshape(-1,1)
        winddirection_ref1 = jnp.tile(wind_direction, sensor_x_ref1.shape[0]).reshape(-1,1)

        # Reflector 2.
        if release_17 == True:
            windspeeds_ref2 = jnp.tile(wind_speed[:-31], sensor_x_ref2.shape[0]).reshape(-1,1)
            winddirection_ref2 = jnp.tile(wind_direction[:-31], sensor_x_ref2.shape[0]).reshape(-1,1)
        elif release_17 == False:
            windspeeds_ref2 = jnp.tile(wind_speed, sensor_x_ref2.shape[0]).reshape(-1,1)
            winddirection_ref2 = jnp.tile(wind_direction, sensor_x_ref2.shape[0]).reshape(-1,1)
        
        # Reflector 3.
        windspeeds_ref3 = jnp.tile(wind_speed, sensor_x_ref3.shape[0]).reshape(-1,1)
        winddirection_ref3 = jnp.tile(wind_direction, sensor_x_ref3.shape[0]).reshape(-1,1)

        # Reflector 4.
        windspeeds_ref4 = jnp.tile(wind_speed, sensor_x_ref4.shape[0]).reshape(-1,1)
        winddirection_ref4 = jnp.tile(wind_direction, sensor_x_ref4.shape[0]).reshape(-1,1)

        # Reflector 5.
        windspeeds_ref5 = jnp.tile(wind_speed, sensor_x_ref5.shape[0]).reshape(-1,1)
        winddirection_ref5 = jnp.tile(wind_direction, sensor_x_ref5.shape[0]).reshape(-1,1)

        # Reflector 6.
        windspeeds_ref6 = jnp.tile(wind_speed, sensor_x_ref6.shape[0]).reshape(-1,1)
        winddirection_ref6 = jnp.tile(wind_direction, sensor_x_ref6.shape[0]).reshape(-1,1)

        # Reflector 7.
        windspeeds_ref7 = jnp.tile(wind_speed, sensor_x_ref7.shape[0]).reshape(-1,1)
        winddirection_ref7 = jnp.tile(wind_direction, sensor_x_ref7.shape[0]).reshape(-1,1)

        # Temporal sensor locations.

        # Reflector 1.
        temporal_sensor_x_ref1 = self.structure_to_vectorise(sensor_x_ref1, sensor_y_ref1, sensor_z_ref1, number_of_time_steps)[1].reshape(-1,1)
        temporal_sensor_y_ref1 = self.structure_to_vectorise(sensor_x_ref1, sensor_y_ref1, sensor_z_ref1, number_of_time_steps)[2].reshape(-1,1)
        temporal_sensor_z_ref1 = self.structure_to_vectorise(sensor_x_ref1, sensor_y_ref1, sensor_z_ref1, number_of_time_steps)[3].reshape(-1,1)

        # Reflector 2.
        if release_17 == True:
            temporal_sensor_x_ref2 = self.structure_to_vectorise(sensor_x_ref2, sensor_y_ref2, sensor_z_ref2, number_of_time_steps - 31)[1].reshape(-1,1)
            temporal_sensor_y_ref2 = self.structure_to_vectorise(sensor_x_ref2, sensor_y_ref2, sensor_z_ref2, number_of_time_steps - 31)[2].reshape(-1,1)
            temporal_sensor_z_ref2 = self.structure_to_vectorise(sensor_x_ref2, sensor_y_ref2, sensor_z_ref2, number_of_time_steps - 31)[3].reshape(-1,1)
        elif release_17 == False:
            temporal_sensor_x_ref2 = self.structure_to_vectorise(sensor_x_ref2, sensor_y_ref2, sensor_z_ref2, number_of_time_steps)[1].reshape(-1,1)
            temporal_sensor_y_ref2 = self.structure_to_vectorise(sensor_x_ref2, sensor_y_ref2, sensor_z_ref2, number_of_time_steps)[2].reshape(-1,1)
            temporal_sensor_z_ref2 = self.structure_to_vectorise(sensor_x_ref2, sensor_y_ref2, sensor_z_ref2, number_of_time_steps)[3].reshape(-1,1)

        # Reflector 3.
        temporal_sensor_x_ref3 = self.structure_to_vectorise(sensor_x_ref3, sensor_y_ref3, sensor_z_ref3, number_of_time_steps)[1].reshape(-1,1)
        temporal_sensor_y_ref3 = self.structure_to_vectorise(sensor_x_ref3, sensor_y_ref3, sensor_z_ref3, number_of_time_steps)[2].reshape(-1,1)
        temporal_sensor_z_ref3 = self.structure_to_vectorise(sensor_x_ref3, sensor_y_ref3, sensor_z_ref3, number_of_time_steps)[3].reshape(-1,1)

        # Reflector 4.
        temporal_sensor_x_ref4 = self.structure_to_vectorise(sensor_x_ref4, sensor_y_ref4, sensor_z_ref4, number_of_time_steps)[1].reshape(-1,1)
        temporal_sensor_y_ref4 = self.structure_to_vectorise(sensor_x_ref4, sensor_y_ref4, sensor_z_ref4, number_of_time_steps)[2].reshape(-1,1)
        temporal_sensor_z_ref4 = self.structure_to_vectorise(sensor_x_ref4, sensor_y_ref4, sensor_z_ref4, number_of_time_steps)[3].reshape(-1,1)

        # Reflector 5.
        temporal_sensor_x_ref5 = self.structure_to_vectorise(sensor_x_ref5, sensor_y_ref5, sensor_z_ref5, number_of_time_steps)[1].reshape(-1,1)
        temporal_sensor_y_ref5 = self.structure_to_vectorise(sensor_x_ref5, sensor_y_ref5, sensor_z_ref5, number_of_time_steps)[2].reshape(-1,1)
        temporal_sensor_z_ref5 = self.structure_to_vectorise(sensor_x_ref5, sensor_y_ref5, sensor_z_ref5, number_of_time_steps)[3].reshape(-1,1)

        # Reflector 6.
        temporal_sensor_x_ref6 = self.structure_to_vectorise(sensor_x_ref6, sensor_y_ref6, sensor_z_ref6, number_of_time_steps)[1].reshape(-1,1)
        temporal_sensor_y_ref6 = self.structure_to_vectorise(sensor_x_ref6, sensor_y_ref6, sensor_z_ref6, number_of_time_steps)[2].reshape(-1,1)
        temporal_sensor_z_ref6 = self.structure_to_vectorise(sensor_x_ref6, sensor_y_ref6, sensor_z_ref6, number_of_time_steps)[3].reshape(-1,1)

        # Reflector 7.
        temporal_sensor_x_ref7 = self.structure_to_vectorise(sensor_x_ref7, sensor_y_ref7, sensor_z_ref7, number_of_time_steps)[1].reshape(-1,1)
        temporal_sensor_y_ref7 = self.structure_to_vectorise(sensor_x_ref7, sensor_y_ref7, sensor_z_ref7, number_of_time_steps)[2].reshape(-1,1)
        temporal_sensor_z_ref7 = self.structure_to_vectorise(sensor_x_ref7, sensor_y_ref7, sensor_z_ref7, number_of_time_steps)[3].reshape(-1,1)
        

        # Tangent of the rolling standard deviation of the horizontal and vertical wind direction.
        if release_17 == True:
            # Horizontal
            horizontal_temporal_tangamma_ts_ref1 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref1.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref2 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref2.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref3 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref3.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref4 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref4.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref5 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref5.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref6 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref6.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref7 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref7.shape[0]).reshape(-1,1)
            # Vertical    
            vertical_temporal_tangamma_ts_ref1 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref1.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref2 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref2.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref3 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref3.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref4 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref4.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref5 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref5.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref6 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref6.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref7 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref7.shape[0]).reshape(-1,1)
        elif source3and4 == True and release_17 == False:
            # Horizontal
            horizontal_temporal_tangamma_ts_ref1 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref1.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref2 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref2.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref3 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref3.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref4 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref4.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref5 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref5.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref6 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref6.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref7 = jnp.tile(horizontal_tangamma.flatten()[:-31], sensor_x_ref7.shape[0]).reshape(-1,1)
            # Vertical
            vertical_temporal_tangamma_ts_ref1 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref1.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref2 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref2.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref3 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref3.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref4 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref4.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref5 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref5.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref6 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref6.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref7 = jnp.tile(vertical_tangamma.flatten()[:-31], sensor_x_ref7.shape[0]).reshape(-1,1)
        else:
            # Horizontal
            horizontal_temporal_tangamma_ts_ref1 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref1.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref2 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref2.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref3 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref3.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref4 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref4.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref5 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref5.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref6 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref6.shape[0]).reshape(-1,1)
            horizontal_temporal_tangamma_ts_ref7 = jnp.tile(horizontal_tangamma.flatten(), sensor_x_ref7.shape[0]).reshape(-1,1)
            # Vertical
            vertical_temporal_tangamma_ts_ref1 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref1.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref2 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref2.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref3 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref3.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref4 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref4.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref5 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref5.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref6 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref6.shape[0]).reshape(-1,1)
            vertical_temporal_tangamma_ts_ref7 = jnp.tile(vertical_tangamma.flatten(), sensor_x_ref7.shape[0]).reshape(-1,1)

        # Atmospheric boundary layer height, source height.
        max_abl = self.atmospheric_state.max_abl
        height = self.source_location.source_location_z

        return windspeeds_ref1, windspeeds_ref2, windspeeds_ref3, windspeeds_ref4, windspeeds_ref5, windspeeds_ref6, windspeeds_ref7, \
            winddirection_ref1, winddirection_ref2, winddirection_ref3, winddirection_ref4, winddirection_ref5, winddirection_ref6, winddirection_ref7, \
            temporal_sensor_x_ref1, temporal_sensor_y_ref1, temporal_sensor_z_ref1, \
            temporal_sensor_x_ref2, temporal_sensor_y_ref2, temporal_sensor_z_ref2, \
            temporal_sensor_x_ref3, temporal_sensor_y_ref3, temporal_sensor_z_ref3, \
            temporal_sensor_x_ref4, temporal_sensor_y_ref4, temporal_sensor_z_ref4, \
            temporal_sensor_x_ref5, temporal_sensor_y_ref5, temporal_sensor_z_ref5, \
            temporal_sensor_x_ref6, temporal_sensor_y_ref6, temporal_sensor_z_ref6, \
            temporal_sensor_x_ref7, temporal_sensor_y_ref7, temporal_sensor_z_ref7, \
            max_abl, height, \
            horizontal_temporal_tangamma_ts_ref1, horizontal_temporal_tangamma_ts_ref2, horizontal_temporal_tangamma_ts_ref3, horizontal_temporal_tangamma_ts_ref4, horizontal_temporal_tangamma_ts_ref5, horizontal_temporal_tangamma_ts_ref6, horizontal_temporal_tangamma_ts_ref7, \
            vertical_temporal_tangamma_ts_ref1, vertical_temporal_tangamma_ts_ref2, vertical_temporal_tangamma_ts_ref3, vertical_temporal_tangamma_ts_ref4, vertical_temporal_tangamma_ts_ref5, vertical_temporal_tangamma_ts_ref6, vertical_temporal_tangamma_ts_ref7
    
    



    def wind_speed_plot(self, save = False, format = "pdf"):
        """
        Plot the OU process simulated wind speeds.
        
        Args:
            save (bool): True if saving the plot.
            format (str): Format to save the plot in.

        Returns:
            (plt.plot): Plot of the OU process simulated wind speeds.
        """
        speeds = self.wind_speed()[:self.wind_field.number_of_time_steps]
        plt.figure(figsize=(15, 4))
        plt.plot(speeds)
        plt.xlabel("Time (s)")
        plt.ylabel("Wind speed (m/s)")
        plt.title("Wind speed over time")
        if save == True:
            plt.savefig("Wind speed over time." + format, dpi=300, bbox_inches="tight")

        return plt.show()



    def wind_direction_plot(self, save = False, format = "pdf"):
        """
        Plot the OU process simulated wind directions.
        
        Args:
            save (bool): True if saving the plot.
            format (str): Format to save the plot in.
            
        Returns:
            (plt.plot): Plot of the OU process simulated wind directions.
        """
        direction = self.wind_direction()[:self.wind_field.number_of_time_steps]
        plt.figure(figsize=(15, 4))
        plt.plot(direction)
        plt.xlabel("Time (s)")
        plt.ylabel("Wind direction (degrees)")
        plt.title("Wind direction over time")
        if save == True:
            plt.savefig("Wind direction over time." + format, dpi=300, bbox_inches="tight")

        return plt.show()













class BackgroundGas(Grid, SourceLocation, AtmosphericState):
    """
    Creates and plots background gas concentration using Gaussian random fields.
    """

    def __init__(self, grid, source_location, atmospheric_state):
        
        self.grid = grid
        self.source_location = source_location
        self.atmospheric_state = atmospheric_state


    def mean_and_std_correction(self, field):
        """
        Corrects the mean and standard deviation of the field to match the simulated background mean and standard deviation.
            
        Args:
            field (Array[float]): Two-dimensional Gaussian field.

        Returns:
            rescaled_field (Array[float]): Two-dimensional Gaussian field with corrected mean and standard deviation.
        """
        rescaled_field = field - np.mean(field)
        rescaled_field = rescaled_field/np.std(rescaled_field)
        rescaled_field = rescaled_field*self.atmospheric_state.background_std + self.atmospheric_state.background_mean
        
        return rescaled_field


    def Gaussian_filtered_Gaussian_random_field(self, kernel_std):
        """ 
        Returns a Gaussian random field with Gaussian filter to the field.
        
        Args:
            kernel_std (float): Standard deviation of the Gaussian filter.
            
        Returns:
            corrected_field (Array[float]):  Random Gaussian samples with corrected mean and standard deviation.
            corrected_smoothed_field (Array[float]): Two-dimensional Gaussian filtered Gaussian field with corrected mean and standard deviation.
        """
        
        # Generate the Gaussian random field
        np.random.seed(self.atmospheric_state.background_seed)
        gaussian_random_field = np.random.normal(loc=self.atmospheric_state.background_mean, scale=self.atmospheric_state.background_std, size=(len(self.grid.x), len(self.grid.y)))
        
        # Apply the Gaussian filter to the field
        smoothed_GRF = gaussian_filter(gaussian_random_field, sigma=kernel_std)
        
        # Correct the mean and standard deviation of the field to reflect simulated background mean and standard deviation
        corrected_field, corrected_smoothed_field = self.mean_and_std_correction(gaussian_random_field), self.mean_and_std_correction(smoothed_GRF)
        
        return corrected_field, corrected_smoothed_field


    def power_law_filtered_Gaussian_random_field(self):
        """ 
        Returns a Gaussian random field with power-law filter to the field using the inverse Fourier transform.
            
        We compute the power spectrum of the field using the Fourier transform. We first compute the 2D frequencies
        using NumPy's "fftfreq" function, and then compute the magnitude of the frequencies raised to the power of
        the exponent of the power-law filter. We add a small value epsilon (in this example, 1e-6) to the spectrum
        before raising it to the power of the exponent. This avoids division by zero errors or the creation of infinite
        values. We also modify the exponent to be a negative value, since we want the power-law filter to reduce the
        power of high-frequency components. We then set the DC component to zero to avoid division by zero errors. We
        then multiply the Fourier transform of the field by the power spectrum and compute the inverse Fourier transform
        to obtain the filtered field. Finally, we return both the original and filtered fields

        Returns:
            corrected_field (Array[float]):  Random Gaussian samples with corrected mean and standard deviation.
            corrected_filtered_field (Array[float]): Two-dimensional power-law filtered Gaussian field with corrected mean and standard deviation.
        """
        
        # Generate the Gaussian random field
        np.random.seed(self.atmospheric_state.background_seed)
        field = np.random.normal(loc=self.atmospheric_state.background_mean, scale=self.atmospheric_state.background_std, size=(len(self.grid.y), len(self.grid.x)))

        # Compute the power spectrum of the field using the Fourier transform
        field_fft = np.fft.fft2(field)
        freqs_x = np.fft.fftfreq(len(self.grid.x), self.grid.dx)
        freqs_y = np.fft.fftfreq(len(self.grid.y.tolist()), self.grid.dy)
        freqs_2d = np.meshgrid(freqs_x, freqs_y)
        freqs = np.sqrt(freqs_2d[0]**2 + freqs_2d[1]**2)
        exponent = -1.5
        epsilon = 1e-6
        freqs = (freqs + epsilon) ** exponent
        freqs[len(self.grid.y.tolist())//2,len(self.grid.x.tolist())//2] = 0 # Remove the DC component

        # Apply the power-law filter to the field using the inverse Fourier transform
        filtered_field_fft = field_fft * freqs
        filtered_field = np.fft.ifft2(filtered_field_fft).real

        # Correct the mean and standard deviation of the field to reflect simulated background mean and standard deviation
        corrected_field, corrected_filtered_field = self.mean_and_std_correction(field), self.mean_and_std_correction(filtered_field)
        
        return corrected_field, corrected_filtered_field


    def background_plot(self, save: bool = False, format: str = "pdf") :
        """ 
        Plots the spatially smoothed background gas concentration and the original random Gaussian samples. 
        
        Args:
            save (bool): True if saving the plot.
            format (str): Format to save the plot in.

        Returns:
            (plt.plot): Heatmap of the spatially smoothed background gas concentration and the original random Gaussian samples.
        """
        
        # Generate the Background Gaussian random field
        filter = self.atmospheric_state.background_filter
        if filter == "Gaussian":
            field, filtered_field = self.Gaussian_filtered_Gaussian_random_field()
        elif filter == "power_law":
            field, filtered_field = self.power_law_filtered_Gaussian_random_field()
        else:
            raise ValueError("Filter must be either 'Gaussian' or 'power_law'.")
        
        # Plot the field
        df1 = pd.DataFrame(field).T
        df2 = pd.DataFrame(filtered_field).T
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plotting the original random Gaussian samples
        if len(self.grid.x) > 10:
            axs[0] = sns.heatmap(df1, ax= axs[0], cmap="viridis", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
            axs[0].set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
            axs[0].set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
            axs[0].set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + 3*self.grid.dy, self.grid.y_range[1]/10)], rotation=0)
            axs[0].set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + 3*self.grid.dx, self.grid.x_range[1]/10)], rotation=0)
        else:
            axs[0] = sns.heatmap(df1, ax= axs[0], cmap="viridis")
            axs[0].set_yticklabels(self.grid.y)
            axs[0].set_xticklabels(self.grid.x)
        axs[0].invert_yaxis()
        axs[0].scatter(float(self.source_location.source_location_x/self.grid.dx) - (self.grid.x_range[0]/self.grid.dx), float(self.source_location.source_location_y/self.grid.dy)-(self.grid.y_range[0]/self.grid.dy), marker='.', s=100, color='orange')
        axs[0].set_title("Original random Gaussian samples")
        colorbar = axs[0].collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        
        # Plotting the spatially smoothed background gas concentration
        if len(self.grid.x) > 10:
            axs[1] = sns.heatmap(df2, ax= axs[1], cmap="viridis", xticklabels=round(len(self.grid.x)/10), yticklabels=round(len(self.grid.y)/10))
            axs[1].set_xticks(jnp.arange(0, len(self.grid.x) + 1, round(len(self.grid.x)/10)))
            axs[1].set_yticks(jnp.arange(0, len(self.grid.y) + 1, round(len(self.grid.y)/10)))
            axs[1].set_yticklabels([round(i,1) for i in jnp.arange(self.grid.y_range[0], self.grid.y_range[1] + 3*self.grid.dy, self.grid.y_range[1]/10)], rotation=0)
            axs[1].set_xticklabels([round(i,1) for i in jnp.arange(self.grid.x_range[0], self.grid.x_range[1] + 3*self.grid.dx, self.grid.x_range[1]/10)], rotation=0)
        else:
            axs[1] = sns.heatmap(df2, ax= axs[1], cmap="viridis")
            axs[1].set_yticklabels(self.grid.y)
            axs[1].set_xticklabels(self.grid.x)
        axs[1].invert_yaxis()
        axs[1].scatter(float(self.source_location.source_location_x/self.grid.dx), float(self.source_location.source_location_y/self.grid.dy), marker='.', s=100, color='orange')
        axs[1].set_title("Spatially smoothed background gas concentration")
        colorbar = axs[1].collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        
        plt.tight_layout(pad=3.0)
        if save == True:
            plt.savefig("backgrounds plot." + format, dpi=300, bbox_inches="tight")

        return plt.show()














class Sensors(GaussianPlume, BackgroundGas, SensorsSettings):
    """
    Creates and plots sensors' measurements using the Gaussian plume model, background gas concentration, and measurement errors.
    """

    def __init__(self, gaussianplume, backgroundgas, sensors_settings):
        
        self.gaussianplume = gaussianplume
        self.backgroundgas = backgroundgas
        self.sensors_settings = sensors_settings
    
    
    def meters_to_cell_nbr(self, coordx, coordy):
        """ 
        Identify a cell's number on the grid from coordinates. This is used in the grid-based inversion.

        Args:
            coordx (Array[float]): Cell's x-coordinate in meters.
            coordy (Array[float]): Cell's y-coordinate in meters.

        Returns:
            index (Array[int]): Cell's number on the grid.
        """

        column = int(coordx/self.gaussianplume.grid.dx) 
        row = int(coordy/self.gaussianplume.grid.dy)
        index = (column)*len(self.gaussianplume.grid.y) + row

        return index    


    def source_rate(self):
        """
        Creates a source rate vector containing the source emission rates within each cell on the grid.
        This is used in the grid-based inversion.
        
        Returns:
            grid_source_rate (Array[float]): Source rate vector in kg/s. 
        """

        grid_source_rate = np.zeros(int(len(self.gaussianplume.grid.x)) * int(len(self.gaussianplume.grid.y))).reshape(-1,1)
        grid_source_rate[self.meters_to_cell_nbr(self.gaussianplume.source_location.source_location_x, self.gaussianplume.source_location.source_location_y)] = self.gaussianplume.atmospheric_state.emission_rate 
        
        return grid_source_rate


    def measurement_errors(self, beam=False):
        """
        Creates the sensors' measurement error vector.
        
        Args:
            beam (bool): True if beam sensors are used.

        Returns:
            measurement_errors (Array[float]): Measurement error vector in PPM.
        """

        np.random.seed(self.gaussianplume.sensors_settings.measurement_error_seed)
        if beam:
            measurement_errors = np.random.normal(0, np.sqrt(self.gaussianplume.sensors_settings.measurement_error_var), 7 * self.gaussianplume.wind_field.number_of_time_steps)
        else:
            measurement_errors = np.random.normal(0, np.sqrt(self.gaussianplume.sensors_settings.measurement_error_var), self.gaussianplume.sensors_settings.sensor_number * self.gaussianplume.wind_field.number_of_time_steps)

        return measurement_errors
    

    def background_vector(self, chilbolton = False) -> np.ndarray:
        """
        Turns a background concentration two-dimensional array to a vector.
        
        Args:
            chilbolton (bool): True if using the Chilbolton sensor set up.

        Returns:
            background_concentration_vector (Array[float]): Background concentration vector in PPM.
            background_concentration (Array[float]): Background concentration two-dimensional array in PPM.
        """
        
        # Generate the Background Gaussian random field
        filter = self.gaussianplume.atmospheric_state.background_filter
        if filter.lower() == "gaussian":
            background_concentration = self.backgroundgas.Gaussian_filtered_Gaussian_random_field(self.state, kernel_std = self.gaussianplume.atmospheric_state.Gaussian_filter_kernel)[1]
        elif filter.lower() == "power_law":
            background_concentration = self.backgroundgas.power_law_filtered_Gaussian_random_field()[1]
        else:
            raise ValueError("Filter must be either 'Gaussian' or 'power_law'.")
        
        if chilbolton == False:
            background_cencentration_vector = jnp.array([background_concentration.flatten()[j] for j in [self.meters_to_cell_nbr(i[0], i[1]) for i in self.gaussianplume.sensors_settings.sensor_locations]])
        elif chilbolton == True:
            background_cencentration_vector = jnp.array([background_concentration.flatten()[j] for j in [self.meters_to_cell_nbr(i[0], i[1]) for i in [self.gaussianplume.sensors_settings.sensor_locations[i] for i in [89, 254, 364, 609, 819, 964, 1049]]]])
    
        return background_cencentration_vector, background_concentration



    def temporal_sensors_measurements(self, grided = False, beam = True,  specified_wind_direction=None, varying_emission_rate=False, constant_mean=True):
        """
        Creates the temporal sensor measurements vector using the Gaussian plume model, background gas concentration, and measurement errors.

        Args:
            grided (bool): True if using the grid-based inversion.
            beam (bool): True if beam sensors are used.
        
        Returns:
            sensors_measurements (Array[float]): Temporal sensor observations vector in PPM.
            A (Array[float]): Temporal Gaussian plume model-based coupling matrix.
            source_rate (Array[float]): Source rate vector in kg/s.
            measurement_errors (Array[float]): Temporal sensor measurement errors vector in PPM.
            background_concentration (Array[float]): Temporal background gas concentrations vector in PPM.
        """

        # Grid-free inversion.
        if grided == False:
            # Point sensors.
            if beam == False:
                if specified_wind_direction is None:
                    fixed = self.gaussianplume.fixed_objects_of_gridfree_coupling_matrix(wind_speed = None, wind_direction = None, number_of_time_steps = None, constant_mean = constant_mean)
                else:
                    fixed = self.gaussianplume.fixed_objects_of_gridfree_coupling_matrix(wind_speed = None, wind_direction = jnp.tile(specified_wind_direction, len(self.sensors_settings.sensor_locations)).reshape(-1,1), number_of_time_steps = None, constant_mean = constant_mean)
                A = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed)
                source_rate = self.gaussianplume.atmospheric_state.emission_rate
                background_concentration = np.repeat(self.background_vector(chilbolton=False)[0], self.gaussianplume.wind_field.number_of_time_steps).reshape(-1,1) 
                measurement_errors = self.measurement_errors().reshape(-1,1)
            # Chilbolton sensors set-up.
            elif beam == True:
                fixed = self.gaussianplume.fixed_objects_of_gridfree_chilbolton_coupling_matrix(simulation = True, wind_speed=None, wind_direction=None, tangamma_ts=None, number_of_time_steps=self.gaussianplume.wind_field.number_of_time_steps, source3and4 = False, release_17 = False)
                fixed_ref1 = fixed[0], fixed[7], fixed[14], fixed[15], fixed[35], fixed[36], fixed[16], fixed[37], fixed[44]
                fixed_ref2 = fixed[1], fixed[8], fixed[17], fixed[18], fixed[35], fixed[36], fixed[19], fixed[38], fixed[45]
                fixed_ref3 = fixed[2], fixed[9], fixed[20], fixed[21], fixed[35], fixed[36], fixed[22], fixed[39], fixed[46]
                fixed_ref4 = fixed[3], fixed[10], fixed[23], fixed[24], fixed[35], fixed[36], fixed[25], fixed[40], fixed[47]
                fixed_ref5 = fixed[4], fixed[11], fixed[26], fixed[27], fixed[35], fixed[36], fixed[28], fixed[41], fixed[48]
                fixed_ref6 = fixed[5], fixed[12], fixed[29], fixed[30], fixed[35], fixed[36], fixed[31], fixed[42], fixed[49]
                fixed_ref7 = fixed[6], fixed[13], fixed[32], fixed[33], fixed[35], fixed[36], fixed[34], fixed[43], fixed[50]
                coupling_matrix_ref1 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref1)
                coupling_matrix_ref2 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref2)
                coupling_matrix_ref3 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref3)
                coupling_matrix_ref4 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref4)
                coupling_matrix_ref5 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref5)
                coupling_matrix_ref6 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref6)
                coupling_matrix_ref7 = self.gaussianplume.temporal_gridfree_coupling_matrix(fixed_ref7)
                # Coupling matrix: Single source case.
                if coupling_matrix_ref1.shape[1] == 1:
                    reshaped_coupling_matrix_ref1 = coupling_matrix_ref1.reshape(self.gaussianplume.wind_field.number_of_time_steps,18*5, order='F')
                    reshaped_coupling_matrix_ref2 = coupling_matrix_ref2.reshape(self.gaussianplume.wind_field.number_of_time_steps,33*5, order='F')
                    reshaped_coupling_matrix_ref3 = coupling_matrix_ref3.reshape(self.gaussianplume.wind_field.number_of_time_steps,22*5, order='F')
                    reshaped_coupling_matrix_ref4 = coupling_matrix_ref4.reshape(self.gaussianplume.wind_field.number_of_time_steps,49*5, order='F')
                    reshaped_coupling_matrix_ref5 = coupling_matrix_ref5.reshape(self.gaussianplume.wind_field.number_of_time_steps,42*5, order='F')
                    reshaped_coupling_matrix_ref6 = coupling_matrix_ref6.reshape(self.gaussianplume.wind_field.number_of_time_steps,29*5, order='F')
                    reshaped_coupling_matrix_ref7 = coupling_matrix_ref7.reshape(self.gaussianplume.wind_field.number_of_time_steps,17*5, order='F')
                    path_averaged_coupling_matrix_ref1 = reshaped_coupling_matrix_ref1.mean(axis=1)
                    path_averaged_coupling_matrix_ref2 = reshaped_coupling_matrix_ref2.mean(axis=1)
                    path_averaged_coupling_matrix_ref3 = reshaped_coupling_matrix_ref3.mean(axis=1)
                    path_averaged_coupling_matrix_ref4 = reshaped_coupling_matrix_ref4.mean(axis=1)
                    path_averaged_coupling_matrix_ref5 = reshaped_coupling_matrix_ref5.mean(axis=1)
                    path_averaged_coupling_matrix_ref6 = reshaped_coupling_matrix_ref6.mean(axis=1)
                    path_averaged_coupling_matrix_ref7 = reshaped_coupling_matrix_ref7.mean(axis=1)
                    path_averaged_A = [path_averaged_coupling_matrix_ref1, path_averaged_coupling_matrix_ref2, path_averaged_coupling_matrix_ref3, path_averaged_coupling_matrix_ref4, path_averaged_coupling_matrix_ref5, path_averaged_coupling_matrix_ref6, path_averaged_coupling_matrix_ref7]
                    A = jnp.array(path_averaged_A).reshape(-1,1)
                # Coupling matrix: Double source case.
                elif coupling_matrix_ref1.shape[1] == 2:
                    reshaped_coupling_matrix_ref1_src1 = coupling_matrix_ref1[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,18*5, order='F')
                    reshaped_coupling_matrix_ref1_src2 = coupling_matrix_ref1[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,18*5, order='F')
                    reshaped_coupling_matrix_ref2_src1 = coupling_matrix_ref2[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,33*5, order='F')
                    reshaped_coupling_matrix_ref2_src2 = coupling_matrix_ref2[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,33*5, order='F')
                    reshaped_coupling_matrix_ref3_src1 = coupling_matrix_ref3[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,22*5, order='F')
                    reshaped_coupling_matrix_ref3_src2 = coupling_matrix_ref3[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,22*5, order='F')
                    reshaped_coupling_matrix_ref4_src1 = coupling_matrix_ref4[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,49*5, order='F')
                    reshaped_coupling_matrix_ref4_src2 = coupling_matrix_ref4[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,49*5, order='F')
                    reshaped_coupling_matrix_ref5_src1 = coupling_matrix_ref5[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,42*5, order='F')
                    reshaped_coupling_matrix_ref5_src2 = coupling_matrix_ref5[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,42*5, order='F')
                    reshaped_coupling_matrix_ref6_src1 = coupling_matrix_ref6[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,29*5, order='F')
                    reshaped_coupling_matrix_ref6_src2 = coupling_matrix_ref6[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,29*5, order='F')
                    reshaped_coupling_matrix_ref7_src1 = coupling_matrix_ref7[:,0].reshape(self.gaussianplume.wind_field.number_of_time_steps,17*5, order='F')
                    reshaped_coupling_matrix_ref7_src2 = coupling_matrix_ref7[:,1].reshape(self.gaussianplume.wind_field.number_of_time_steps,17*5, order='F')
                    path_averaged_coupling_matrix_ref1_src1 = reshaped_coupling_matrix_ref1_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref1_src2 = reshaped_coupling_matrix_ref1_src2.mean(axis=1)
                    path_averaged_coupling_matrix_ref2_src1 = reshaped_coupling_matrix_ref2_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref2_src2 = reshaped_coupling_matrix_ref2_src2.mean(axis=1)
                    path_averaged_coupling_matrix_ref3_src1 = reshaped_coupling_matrix_ref3_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref3_src2 = reshaped_coupling_matrix_ref3_src2.mean(axis=1)
                    path_averaged_coupling_matrix_ref4_src1 = reshaped_coupling_matrix_ref4_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref4_src2 = reshaped_coupling_matrix_ref4_src2.mean(axis=1)
                    path_averaged_coupling_matrix_ref5_src1 = reshaped_coupling_matrix_ref5_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref5_src2 = reshaped_coupling_matrix_ref5_src2.mean(axis=1)
                    path_averaged_coupling_matrix_ref6_src1 = reshaped_coupling_matrix_ref6_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref6_src2 = reshaped_coupling_matrix_ref6_src2.mean(axis=1)
                    path_averaged_coupling_matrix_ref7_src1 = reshaped_coupling_matrix_ref7_src1.mean(axis=1)
                    path_averaged_coupling_matrix_ref7_src2 = reshaped_coupling_matrix_ref7_src2.mean(axis=1)
                    source_1_path_averaged_A = [path_averaged_coupling_matrix_ref1_src1, path_averaged_coupling_matrix_ref2_src1, path_averaged_coupling_matrix_ref3_src1, path_averaged_coupling_matrix_ref4_src1, path_averaged_coupling_matrix_ref5_src1, path_averaged_coupling_matrix_ref6_src1, path_averaged_coupling_matrix_ref7_src1]
                    source_2_path_averaged_A = [path_averaged_coupling_matrix_ref1_src2, path_averaged_coupling_matrix_ref2_src2, path_averaged_coupling_matrix_ref3_src2, path_averaged_coupling_matrix_ref4_src2, path_averaged_coupling_matrix_ref5_src2, path_averaged_coupling_matrix_ref6_src2, path_averaged_coupling_matrix_ref7_src2]
                    source_1_A = jnp.array(source_1_path_averaged_A).reshape(-1,1)
                    source_2_A = jnp.array(source_2_path_averaged_A).reshape(-1,1)
                    A = jnp.concatenate((source_1_A, source_2_A), axis=1)
                # Source_rate, background_concentration, measurement_errors.
                source_rate = self.gaussianplume.atmospheric_state.emission_rate 
                background_concentration = np.tile(self.background_vector(chilbolton=True)[0], self.gaussianplume.wind_field.number_of_time_steps).reshape(-1,1) 
                measurement_errors = self.measurement_errors(beam=True).reshape(-1,1)
        
        # Grid-based inversion.
        elif grided == True:
            fixed = self.gaussianplume.fixed_objects_of_grided_coupling_matrix()
            A = self.gaussianplume.temporal_grided_coupling_matrix(fixed)
            source_rate = self.source_rate()
            background_concentration = np.repeat(self.background_vector(chilbolton=False)[0], self.gaussianplume.wind_field.number_of_time_steps).reshape(-1,1)
            measurement_errors = self.measurement_errors().reshape(-1,1)
        if varying_emission_rate == False:
            sensors_measurements =  np.matmul(A, source_rate.reshape(-1,1)) + background_concentration + measurement_errors
        elif varying_emission_rate == True:
            sensors_measurements =  (A.flatten() * jnp.tile(source_rate,36)).reshape(-1,1) + background_concentration + measurement_errors  
        
        return sensors_measurements.reshape(-1,1), A, source_rate, measurement_errors, background_concentration


    def atmospheric_methane_and_sensors(self, save = False, chilbolton = False, format = "pdf", angles=False):
        """
        Plots the Gaussian plume, background gas concentration, and sensors.

        Args:
            save (bool): True if saving the plot.
            chilbolton (bool): True if using the Chilbolton sensor set up.
            format (str): Format to save the plot in.
            angles (bool): True if plotting lines from the source representing the wind direction.

        Returns:
            (plt.plot): Heatmap of the Gaussian plume, background gas concentration, and sensors.
        """
        
        # Sensor locations and background concentration.
        sensors = self.gaussianplume.sensors_settings.sensor_locations
        background_concentration = self.background_vector(chilbolton)[1]

        # Gaussian plume model concentrations for each source.
        concentrations = 0
        for source_nbr in range(len(self.gaussianplume.source_location.source_location_x)):
            concentrations += self.gaussianplume.gaussian_plume_for_plot(source_nbr)
        df = pd.DataFrame( (concentrations.reshape(len(self.gaussianplume.grid.y), len(self.gaussianplume.grid.x)) + background_concentration))
        
        # Plotting.
        plt.figure(figsize=(10, 10))
        colors = cm.plasma(np.linspace(0.40, 1, len(sensors)))
        plt.figure(figsize=(7, 7))

        # Heatmap.
        if (len(self.gaussianplume.grid.x) > 10) and (len(self.gaussianplume.grid.y) > 10):
            ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.gaussianplume.grid.x)/10), yticklabels=round(len(self.gaussianplume.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.gaussianplume.grid.x) + 1, round(len(self.gaussianplume.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.gaussianplume.grid.y) + 1, round(len(self.gaussianplume.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.y_range[0], self.gaussianplume.grid.y_range[1] + 3*self.gaussianplume.grid.dy, self.gaussianplume.grid.y_range[1]/10)], rotation=0, fontsize=22)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.x_range[0], self.gaussianplume.grid.x_range[1] + 3*self.gaussianplume.grid.dx, self.gaussianplume.grid.x_range[1]/10)], rotation=45, fontsize=22)
        else:
            ax = sns.heatmap(df, cmap="jet")
            ax.set_yticklabels(self.gaussianplume.grid.y)
            ax.set_xticklabels(self.gaussianplume.grid.x)
        ax.invert_yaxis()

        # Plotting source and sensor locations.
        for source_nbr in range(len(self.gaussianplume.source_location.source_location_x)):
            ax.scatter(float(self.gaussianplume.source_location.source_location_x[source_nbr]/self.gaussianplume.grid.dx), float(self.gaussianplume.source_location.source_location_y[source_nbr]/self.gaussianplume.grid.dy), marker='.', s=100, color='orange')
        for i in range(len(sensors)):
            ax.scatter(float(sensors[i][0]/self.gaussianplume.grid.dx), float(sensors[i][1]/self.gaussianplume.grid.dy), marker='*', s=50, color=colors[i])
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Concentration (PPM)', fontsize=22)
        colorbar.ax.tick_params(labelsize=22)

        # Plotting wind direction.
        if angles:
            degrees = [5, 10, 20, 40, 70, 100, 130, 160, 190, 360]
            cmap = plt.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, len(degrees))]
            # Calculate the maximum distance from the origin to any of the points
            max_distance = 700
            for degree, color in zip(degrees, colors):
                # Calculate the end points of the lines
                end_point1 = [float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx) + max_distance*np.cos(np.radians(degree / 2.0)), float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy) + max_distance*np.sin(np.radians(degree / 2.0))]
                end_point2 = [float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx) + max_distance*np.cos(np.radians(-degree / 2.0)), float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy) + max_distance*np.sin(np.radians(-degree / 2.0))]

                # Plot the lines with the same color
                ax.plot([float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx), end_point1[0]], [float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy), end_point1[1]], color=color, label=f'{degree} degrees')
                ax.plot([float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx), end_point2[0]], [float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy), end_point2[1]], color=color)
            # Set the aspect of the plot to equal to ensure the angles are correct
            ax.set_aspect('equal')
            ax.legend()
        
        # Plot labels.
        plt.xlabel("$x$: location (m)", fontsize=22)
        plt.ylabel("$y$: location (m)", fontsize=22)
        if save == True:
            plt.savefig("Gaussian plume, background and sensors." + format, dpi=300, bbox_inches="tight")

        return plt.show()
    

    def log_atmospheric_methane_and_sensors(self, save = False, chilbolton=False, format = "pdf"):
        """
        Plots the log Gaussian plume, background gas concentration, and sensors.

        Args:
            save (bool): True if saving the plot.
            chilbolton (bool): True if using the Chilbolton sensor set up.
            format (str): Format to save the plot in.

        Returns:
            (plt.plot): Heatmap of the log Gaussian plume, background gas concentration, and sensors.
        """

        # Sensor locations and background concentration.
        sensors = self.gaussianplume.sensors_settings.sensor_locations
        background_concentration = self.background_vector(chilbolton)[1]

        # Gaussian plume model concentrations for each source.
        concentrations = 0
        for source_nbr in range(len(self.gaussianplume.source_location.source_location_x)):
            concentrations += self.gaussianplume.gaussian_plume_for_plot(source_nbr)
        df = pd.DataFrame( np.log(concentrations.reshape(len(self.gaussianplume.grid.y), len(self.gaussianplume.grid.x)) + background_concentration))
        
        # Plotting.
        plt.figure(figsize=(15, 10))
        
        # Heatmap.
        if (len(self.gaussianplume.grid.x) > 10) and (len(self.gaussianplume.grid.y) > 10):
            ax = sns.heatmap(df, cmap="jet", xticklabels=round(len(self.gaussianplume.grid.x)/10), yticklabels=round(len(self.gaussianplume.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.gaussianplume.grid.x) + 1, round(len(self.gaussianplume.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.gaussianplume.grid.y) + 1, round(len(self.gaussianplume.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.y_range[0], self.gaussianplume.grid.y_range[1] + self.gaussianplume.grid.dy + 1, self.gaussianplume.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.x_range[0], self.gaussianplume.grid.x_range[1] + self.gaussianplume.grid.dx + 1, self.gaussianplume.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="jet")
            ax.set_yticklabels(self.gaussianplume.grid.y)
            ax.set_xticklabels(self.gaussianplume.grid.x)
        ax.invert_yaxis()

        # Plotting source and sensor locations.
        for source_nbr in range(len(self.gaussianplume.source_location.source_location_x)):
            ax.scatter(float(self.gaussianplume.source_location.source_location_x[source_nbr]/self.gaussianplume.grid.dx), float(self.gaussianplume.source_location.source_location_y[source_nbr]/self.gaussianplume.grid.dy),
                    marker='.', s=100, color='orange')
        for i in range(len(sensors)):
            ax.scatter(float(sensors[i][0]/self.gaussianplume.grid.dx), float(sensors[i][1]/self.gaussianplume.grid.dy), marker='*', s=50, color='white')

        # Plot labels.
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Log parts per million (PPM)')
        plt.title("Log Gaussian plume, background and sensors")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Log Gaussian plume, background and sensors." + format, dpi=300, bbox_inches="tight")

        return plt.show()


    def plot_polar_graph(self, save):
        plt.style.use('default')
        degrees_list = [self.gaussianplume.wind_direction() for i in range(self.sensors_settings.sensor_number)]
        data = self.temporal_sensors_measurements(grided=False, beam=False)[0]
        concentrations_list = [data[self.gaussianplume.wind_field.number_of_time_steps*(i-1):self.gaussianplume.wind_field.number_of_time_steps*i] for i in range(1, self.sensors_settings.sensor_number+1)]
        # Create a polar plot
        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        # Create a color map
        colors = cm.plasma(np.linspace(0.40, 1, self.sensors_settings.sensor_number))
        # Plot each line
        for degrees, concentrations, color in zip(degrees_list, concentrations_list, colors):
            # Convert degrees to radians
            radians = jnp.deg2rad(degrees)
            ax.plot(radians, concentrations, color=color, lw=2.5)  # Plot the line
            ax.fill(radians, concentrations, alpha=0.0, color=color)  # Fill the area under the line

        # Set the title and labels
        ax.set_xlabel("Degrees", color='Black', fontsize=22)
        ax.set_ylabel("Concentration (PPM)", color='Black', fontsize=22)

        # Set the concentration plotting limits
        min_concentration = min(min(concentrations) for concentrations in concentrations_list)
        max_concentration = max(max(concentrations) for concentrations in concentrations_list)
        ax.set_ylim([1.95, max_concentration])
        # Create a scalar mappable object
        sm = cm.ScalarMappable(cmap = cm.get_cmap('plasma'), norm=plt.Normalize(vmin=min_concentration, vmax=max_concentration))
        sm.set_array([])
            # Get the labels
        labels = ax.get_xticklabels()

        # Loop over the labels
        for label in labels:
            # Get the text of the label
            text = label.get_text()

            # If the text is '180', hide the label
            if text == '180°':
                label.set_visible(False)
            
            label.set_fontsize(20)  # change '14' to your desired font size

        # Add a colorbar
        if save == True:
            plt.savefig("polar_plot.pdf", dpi=300, bbox_inches='tight', transparent=False)

        # Show the plot
        plt.show()