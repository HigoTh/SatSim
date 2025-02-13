import numpy as np
from enum import Enum, IntEnum
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from typing import List
from scipy.optimize import fsolve
import geom as ge
from mpl_toolkits.basemap import Basemap
import plotly.graph_objects as go
from itertools import chain
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# ---------------------------------- Auxiliary functions ----------------------
# -----------------------------------------------------------------------------

def sort_coordinates(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]

def sph2cart(r, el, az):
    return np.array( [ r * np.sin(el) * np.cos(az), 
                       r * np.sin(el) * np.sin(az),
                       r * np.cos(el) ] )

def azel2latlong( az_rad, el_rad ):
    long = az_rad - np.pi if az_rad > np.pi else az_rad
    lat = -el_rad + (np.pi/2)
    return lat, long

def latlong2azel( lat, long ):
    az_rad = long
    el_rad = -lat + (np.pi/2)
    return az_rad, el_rad

def cart_2_sph( xyz ):
    
    # Spherical matrix
    sph = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    # Radius
    sph[:,0] = np.sqrt(xy + xyz[:,2]**2)
    # Elevation
    sph[:,1] = np.arctan2( np.sqrt(xy), xyz[:,2] )
    # Azimuth
    sph[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return sph

def cart_2_ll( xyz ):
    
    # Spherical matrix
    sph = np.zeros([max(xyz.shape),2])
    xy = xyz[:,0]**2 + xyz[:,1]**2
    # Latitude
    sph[:,1] = -np.arctan2( np.sqrt(xy), xyz[:,2] ) + (np.pi/2)
    # Longitude
    sph[:,0] = np.arctan2(xyz[:,1], xyz[:,0])
    return sph

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

# -----------------------------------------------------------------------------
# ---------------------------------- CLASSES ----------------------------------
# -----------------------------------------------------------------------------

# Physical constants ----------------------------------------------------------
class PhyConstants(Enum):
    EARTH_RAD_KM = 6378.145
    EARTH_ANG_RATE_ROT_DG = 4.1780745823 * 1e-3
    EARTH_ANG_RATE_ROT_RAD = np.deg2rad( EARTH_ANG_RATE_ROT_DG )
    EARTH_NON_SPH = 0.001082636
    GEO_ORBIT_RAD_KM = 42164.2
    GRAV_CONSTANT = 3.986012*1e5
    LIGHT_SPEED = 2.99792458 * 1e5

# Classes for error handling --------------------------------------------------
class Errors(IntEnum):
    ERROR_INVALID_EARTH_STATIONS_COORDS = 700
    ERROR_INVALID_COV_POLYGON_SHAPE = 701

class ErrorStatus:

    _error_msg_dict = {
        Errors.ERROR_INVALID_EARTH_STATIONS_COORDS: 'Lista de coordenadas de estações terrestres inválida.',
        Errors.ERROR_INVALID_COV_POLYGON_SHAPE: 'Pontos de cobertura não formam um polígono.'
    }

    def __init__(self, status_code: Errors, status_aux_string: str = ""):

        self.status_code = status_code
        error_message = (
            self._error_msg_dict[status_code]
            if not status_aux_string
            else self._error_msg_dict[status_code].format(
                aux_string=status_aux_string
            )
        )
        msg_type = 'Erro' if int( status_code ) >= 700 else 'Alerta'
        self.status_message = f'{msg_type} {int( status_code )}: {error_message}'

class Exceptions(Exception):

    def __init__(self, message, error_code: Errors):
        super().__init__(message)
        self.error_code = error_code

glb_earth_sphere = ge.Sphere( ge.Point( 0,0,0 ), PhyConstants.EARTH_RAD_KM.value )
# -----------------------------------------------------------------------------

class EarthStation:

    def __init__( self, es_initial_coords_lat_lon: list ) -> None:

        self.es_initial_coords_ll = es_initial_coords_lat_lon
        self.es_initial_coords_cart = self._get_cart_coordinates( es_initial_coords_lat_lon )
        self.es_current_coords_ll = self.es_initial_coords_ll
        self.es_current_coords_cart = self.es_initial_coords_cart

    def _get_cart_coordinates( self, es_initial_coords_lat_lon: list ):

        # Convert lat-lon coordinates to cartesian
        es_initial_coords_xyz = sph2cart(PhyConstants.EARTH_RAD_KM.value, es_initial_coords_lat_lon[0], es_initial_coords_lat_lon[1])
        return es_initial_coords_xyz

    def update_position( self, dt: float ):

        self.es_current_coords_ll = [ self.es_current_coords_ll[0], self.es_current_coords_ll[1] + PhyConstants.EARTH_ANG_RATE_ROT_RAD.value * dt ]
        self.es_current_coords_cart = sph2cart(PhyConstants.EARTH_RAD_KM.value, self.es_current_coords_ll[0], self.es_current_coords_ll[1])
        return


class Earth:
    """ Earth reference class.
    """

    # Earth's rotation angular rate
    angular_rate = PhyConstants.EARTH_ANG_RATE_ROT_RAD.value
    # Geostationary satellite orbit radius
    earth_rad = PhyConstants.EARTH_RAD_KM.value

    def __init__(self) -> None:

        # Earth reference sphere
        self.sphere = ge.Sphere( ge.Point( 0,0,0 ), self.earth_rad )
        # Earth reference vectorial basis
        self.vec_basis = self._get_vec_basis()
        # World view bounding box
        self.world_view_poly = Polygon([(-178, -88), (178, -88), (178, 88), (-178, 88), (-178, -88)])
        # Coverage zones
        self.coverage_zones = {}

    def _get_vec_basis( self ):
        """Creates the earth's reference vector base."""
        # Compute basis vectors, x and y determine the plane of the equator 
        # and z points to the north pole.
        x = ge.Vector( [ 1, 0, 0 ] ).norm() 
        y = ge.Vector( [ 0, 1, 0 ] ).norm()
        z = ge.Vector( [ 0, 0, 1 ] ).norm()
        # Get vector basis
        vec_basis = ge.VectorBasis( x, y, z )
        return vec_basis

    def _get_rotation_mat( self, angle: float ):
        """ Compute the rotation matrix.
        Args:
            angle (float): Angle of rotation around z axis.

        Returns:
            array: Rotation matrix.
        """

        # Rotation matrix
        r_1 = [ np.cos( angle ), -np.sin( angle ), 0 ]
        r_2 = [ np.sin( angle ), np.cos( angle ), 0 ]
        r_3 = [ 0, 0, 1 ]
        rot_mat = np.array( [ r_1, r_2, r_3] )
        return rot_mat

    def update_position( self, t: float ):

        # Get rotation matrix
        rot_matrix = self._get_rotation_mat( self.angular_rate * t )
        # Rotate base vectors
        x = ge.Vector( np.matmul( rot_matrix, self.vec_basis.v1.asarray() ) )
        y = ge.Vector( np.matmul( rot_matrix, self.vec_basis.v2.asarray() ) )
        z = self.vec_basis.v3
        self.vec_basis = ge.VectorBasis( x, y, z )
        
        # Rotate points on surface
    
        return

    def plot_earth( self, vecs_exag = 10000, return_fig = False ):
        """ Plots satellite position, base vector, pointing vector and terrestrial sphere.
        Returns:
            Object of the plotted figure.
        """

        fig = plt.figure(figsize=(10,10), dpi=80)
        
        ax = fig.add_subplot(111, projection='3d')
        # Make data
        r = PhyConstants.EARTH_RAD_KM.value
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        ax.plot_surface(x, y, z, color='linen', alpha=0.5)
        # plot circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        ax.plot(x, y, z, color='black', alpha=0.75)
        ax.plot(z, x, y, color='black', alpha=0.75)

        ## add axis lines
        zeros = np.zeros(1000)
        line = np.linspace(-r,r,1000)

        ax.plot(line, zeros, zeros, color='black', alpha=0.75)
        ax.plot(zeros, line, zeros, color='black', alpha=0.75)
        ax.plot(zeros, zeros, line, color='black', alpha=0.75)

        # Plot vector basis
        r = vecs_exag
        origin = [0,0,0]
        ax.quiver( *origin, r * self.vec_basis.v1.dx, r * self.vec_basis.v1.dy, r * self.vec_basis.v1.dz, color='blue')
        ax.quiver( *origin, r * self.vec_basis.v2.dx, r * self.vec_basis.v2.dy, r * self.vec_basis.v2.dz, color='red')
        ax.quiver( *origin, r * self.vec_basis.v3.dx, r * self.vec_basis.v3.dy, r * self.vec_basis.v3.dz, color='black')
        
        ax.set_zlim(-PhyConstants.GEO_ORBIT_RAD_KM.value / 2, PhyConstants.GEO_ORBIT_RAD_KM.value / 2)
        ax.set_ylim(-PhyConstants.GEO_ORBIT_RAD_KM.value / 2, PhyConstants.GEO_ORBIT_RAD_KM.value / 2)
        ax.set_xlim(-PhyConstants.GEO_ORBIT_RAD_KM.value / 2, PhyConstants.GEO_ORBIT_RAD_KM.value / 2)

        ret = fig if return_fig else None
        
        plt.show()

        return ret


class SatAntenna:

    def __init__(self, beam_dir_vector: List[float], hpbw_dg: float ) -> None:
        """Initializes an antenna instance.
        Args:
            beam_dir_vector (List[float]): Beam direction vector.
            hpbw_dg (float): Antenna HPBW expressed in degrees.
        """
        self.beam_direction_vector = ge.Vector( beam_dir_vector ).norm()
        self.hpbw_dg = hpbw_dg
        self.hpbw_rad = np.deg2rad( hpbw_dg )
        self.max_gain = 2.0 - ( 2.0 / np.log2( np.cos( self.hpbw_rad ) ) )
        self.power_factor = ( self.max_gain / 2.0 ) - 1.0
        self.vec_basis = self.get_antenna_vec_basis( beam_dir_vector )

    def ant_gain_on_direction( self, dir_vector: List[float] ):
        """ 
        Calculates antenna gain based on a direction vector.
        Args:
            dir_vector (List[float]): Direction vector.

        Returns:
            float: Antenna gain.
        """
        cos_th = self.beam_direction_vector.dot( ge.Vector( dir_vector ).norm() )
        ant_gain = 0.0 if cos_th < 0.0 else ( self.max_gain * ( cos_th**( self.power_factor ) ) )
        return ant_gain

    def ant_gain_on_angle( self, dir_angle: float ):
        """ 
        Calculates antenna gain based on a relative angle.
        Args:
            dir_angle (float): relative angle expressed in radians.

        Returns:
            float: Antenna gain.
        """
        cos_th = np.cos( np.deg2rad( dir_angle ) )
        ant_gain = 0.0 if cos_th < 0.0 else ( self.max_gain * ( cos_th**( self.power_factor ) ) )
        return ant_gain

    def get_antenna_vec_basis( self, beam_dir_vector: List[float] ):
        """Calculates the vector basis of the antenna.
        Returns:
            vec_basis (VecBasis): Vector basis.
        """
        # Compute basis vectors
        r_u = ge.Vector( beam_dir_vector ).norm()
        v_u = ge.Vector( [ -r_u.dy, r_u.dx, 0.0 ] ).norm()
        t_u = r_u.cross( v_u ).norm()

        # Get vector basis
        vec_basis = ge.VectorBasis( t_u, v_u, r_u )
        return vec_basis

    def __str__(self):
        """Minimal string representation in parentheses"""
        return (f"({self.beam_direction_vector}, {self.hpbw_dg}°)")

    def __repr__(self):
        """String representation with class included"""
        return "SatAntenna" + str(self)


class GeoSatellite:
    """Geostationary satellite class.
    """

    # Earth's rotation angular rate
    angular_rate = PhyConstants.EARTH_ANG_RATE_ROT_RAD.value
    # Geostationary satellite orbit radius
    orbit_rad = PhyConstants.GEO_ORBIT_RAD_KM.value

    def __init__(self, sat_name: str, init_az: float, lat_long_target_position: List[float], ant_hpbw: float, tx_power: float ) -> None:

        """Initializing a geo satellite instance.
        Args:
            init_az (float): Initial azimuth angle.
            lat_lon_target_position (List[float]): List with central position (lat(°), lon(°)) 
                on the Earth's surface where the satellite beam is oriented in the initial instant.
            ant_hpbw: HPBW of satellite antenna.
        """
        self.name = sat_name
        # Initial states
        self.initial_sph_el_az = [np.pi/2, init_az] # Elevation-azimuth coordinates
        self.initial_lat_long = azel2latlong( self.initial_sph_el_az[1], self.initial_sph_el_az[0] ) # Lat-Long coordinates
        self.initial_cart = sph2cart( self.orbit_rad, self.initial_sph_el_az[0], self.initial_sph_el_az[1]) # Cartesian coordinates
        # Current states
        self.current_sph_el_az = self.initial_sph_el_az
        self.current_lat_long = self.initial_lat_long
        self.current_cart = self.initial_cart

        # Append to the paths
        self.path_sph_el_az = [ self.current_sph_el_az ]
        self.path_lat_long = [ self.current_lat_long ]
        self.path_cart = [ self.current_cart ]
        self.time = [ 0 ]

        self.tx_power = tx_power
        # Vector basis
        self._sat_vec_basis = self._get_ref_vectorial_base()

        self._init_lat_lon_target_position = lat_long_target_position
        # Beam direction vector (Earth and Satellite VB)
        self._beam_vector_be = self._get_beam_vector_be( lat_long_target_position )
        self._beam_vector_bs = self._get_beam_vector_bs()
        self._beam_dir_el_az = self._get_el_az_angles_beam()

        # Satellite antenna
        self.antenna = SatAntenna( self._beam_vector_be, ant_hpbw )

        # Coverage data
        self.coverage_data = {}

    def _get_ref_vectorial_base( self ):
        """ Determines the satellite's reference vector basis and the basis matrix.
        Returns:
            vec_basis (VecBasis): Vector basis.
        """
        # Compute basis vectors
        r_u = ge.Vector( self.current_cart ).norm()
        v_u = ge.Vector( [ -r_u.dy, r_u.dx, 0.0 ] ).norm()
        t_u = r_u.cross( v_u ).norm()
        # Get vector basis
        vec_basis = ge.VectorBasis( v_u, r_u, t_u )
        return vec_basis

    def _get_beam_vector_be( self, lat_lon_target_position ):
        """ Calculates the beam direction vector referenced in the terrestrial vector basis.
        Returns:
            beam_vector_be (Vector): beam direction vector (Earth vector basis)
        """
        az, el = latlong2azel( lat_lon_target_position[0], lat_lon_target_position[1] )

        # Target point of earth surface
        earth_surface_pos = sph2cart( PhyConstants.EARTH_RAD_KM.value, el, az)
        
        # Beam vector on earth basis
        beam_vector_be = (ge.Vector( earth_surface_pos ) - ge.Vector( self.current_cart )).norm()

        return beam_vector_be

    def _get_beam_vector_bs( self ):
        """ Calculates the beam direction vector referenced in the satellite vector basis.
        Returns:

        """
        # Beam vector on satellite basis
        beam_vector_base_s = np.linalg.inv( self._sat_vec_basis.basis_mat ).dot( self._beam_vector_be.asarray() )
        beam_vector_bs = ge.Vector( beam_vector_base_s ).norm()
        return beam_vector_bs

    def _get_el_az_angles_beam( self ):
        """Calculates the azimuth and elevation angles of the beam direction with 
        respect to the satellite base vector.
        """
        # Azimuth and elevation angles referenced to the satellite basis
        az = np.arctan2( self._beam_vector_bs.dy, self._beam_vector_bs.dx )
        el = np.arccos( self._beam_vector_bs.dz )

        return el, az

    def _update_beam_vector( self ):
        """Updates the beam direction vector referenced in the earth vector basis.
        """
        # Azimuth and elevation angles on the reference basis
        az = self._beam_dir_el_az[1]
        el = self._beam_dir_el_az[0]
        # Update the beam vector 
        cart_p = sph2cart(1.0, el, az)

        self._beam_vector_be = ge.Vector( cart_p[ 0 ] * self._sat_vec_basis.v1 + \
            cart_p[ 1 ] * self._sat_vec_basis.v2 + \
            cart_p[ 2 ] * self._sat_vec_basis.v3 ).norm()

        self.antenna.beam_direction_vector = self._beam_vector_be
        self.antenna.vec_basis = self.antenna.get_antenna_vec_basis( self._beam_vector_be.asarray() )

        return

    def update_position(self, dt:float):
        """ Updates the position, base vector and pointing vector of the satellite.
        Returns:
            True if successful, False otherwise.
        """
        # Update geo satellite position
        self.current_sph_el_az = [ self.current_sph_el_az[0], self.current_sph_el_az[1] + self.angular_rate * dt ]
        self.current_lat_long = azel2latlong( self.current_sph_el_az[1], self.current_sph_el_az[0] )
        self.current_cart = sph2cart( self.orbit_rad, self.current_sph_el_az[0], self.current_sph_el_az[1])

        # Update vectorial base
        self._sat_vec_basis = self._get_ref_vectorial_base()
        # Update beam vector
        self._update_beam_vector()
        # Update path record
        self.path_sph_el_az.append( self.current_sph_el_az )
        self.path_lat_long.append( self.current_lat_long )
        self.path_cart.append( self.current_cart )
        self.time.append( dt )

        return True

    def plot_sat( self, vecs_exag = 10000, return_fig = False ):
        """ Plots satellite position, base vector, pointing vector and terrestrial sphere.
        Returns:
            Object of the plotted figure.
        """

        fig = plt.figure(figsize=(10,10), dpi=80)
        
        ax = fig.add_subplot(111, projection='3d')
        # Make data
        r = PhyConstants.EARTH_RAD_KM.value
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        ax.plot_surface(x, y, z, color='linen', alpha=0.5)
        # plot circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        ax.plot(x, y, z, color='black', alpha=0.75)
        ax.plot(z, x, y, color='black', alpha=0.75)

        ## add axis lines
        zeros = np.zeros(1000)
        line = np.linspace(-r,r,1000)

        ax.plot(line, zeros, zeros, color='black', alpha=0.75)
        ax.plot(zeros, line, zeros, color='black', alpha=0.75)
        ax.plot(zeros, zeros, line, color='black', alpha=0.75)

        # Plot geo satellite
        ax.scatter( *self.current_cart, marker='x', linewidth=4, s=150, color='blue' )
        # Plot vector basis
        r = vecs_exag
        ax.quiver( *self.current_cart, r * self._sat_vec_basis.v1.dx, r * self._sat_vec_basis.v1.dy, r * self._sat_vec_basis.v1.dz, color='blue')
        ax.quiver( *self.current_cart, r * self._sat_vec_basis.v2.dx, r * self._sat_vec_basis.v2.dy, r * self._sat_vec_basis.v2.dz, color='red')
        ax.quiver( *self.current_cart, r * self._sat_vec_basis.v3.dx, r * self._sat_vec_basis.v3.dy, r * self._sat_vec_basis.v3.dz, color='black')
        ax.quiver( *self.current_cart, r * self._beam_vector_be.dx, r * self._beam_vector_be.dy, r * self._beam_vector_be.dz, color='green')

        ax.set_zlim(-PhyConstants.GEO_ORBIT_RAD_KM.value, PhyConstants.GEO_ORBIT_RAD_KM.value)
        ax.set_ylim(-PhyConstants.GEO_ORBIT_RAD_KM.value, PhyConstants.GEO_ORBIT_RAD_KM.value)
        ax.set_xlim(-PhyConstants.GEO_ORBIT_RAD_KM.value, PhyConstants.GEO_ORBIT_RAD_KM.value)

        ret = fig if return_fig else None
        
        plt.show()

        return ret

class NGeoSatellite:

    def __init__(self,
                 sat_name: str,
                 init_asc_node_long: float, 
                 inclination_angle: float,
                 apogee_dist: float,
                 perigee_dist: float,
                 init_perigee_arg: float,
                 init_orbit_angle: float,
                 ant_hpbw: float, 
                 tx_power: float ) -> None:

        self.name = sat_name

        # Constant factors ----------------------------------------------------
        # Inclination angle - i
        self.inclination_angle = inclination_angle
        # Distance from the centre of the Earth to the satellite at apogee - Ra
        self.apogee_dist = apogee_dist
        # Distance from the centre of the Earth to the satellite at perigee - Rp
        self.perigee_dist = perigee_dist
        # Semi-major axis - a
        self.semi_major_axis = (apogee_dist + perigee_dist) / 2.0
        # Eccentricity - e
        self.eccentricity = (apogee_dist - perigee_dist) / (apogee_dist + perigee_dist)
        # Focal parameter - p
        self.focal_parameter = self.semi_major_axis * ( 1 - (self.eccentricity)**2 )
        # Line of nodes - n
        self.line_of_nodes = self._compute_line_of_nodes()
        # Rate of ascending node longitude secular drift - Omega_r
        self.rate_asc = self._compute_rate_asc()
        # Perigee argument secular shift rate - omega_r
        self.perigee_arg_prec = self._compute_perigee_arg_prec()
        # Orbit period - T
        self.orbit_period = (2 * np.pi) * np.sqrt( (self.semi_major_axis**3) / PhyConstants.GRAV_CONSTANT.value )

        # Time-dependent factors ----------------------------------------------
        self.time = [0]
        # Longitude of ascending node
        self.asc_node_long = [init_asc_node_long]
        # Argument of perigee
        self.perigee_arg = [init_perigee_arg]
        # Orbit angle - v
        self.orbit_angle = [init_orbit_angle]
        # Eccentric anomaly - E
        self.ecc_anom = [self._compute_init_ecc_anomaly()]
        # Mean anomaly - M
        self.mean_anom = [ self.ecc_anom[-1] - self.eccentricity * np.sin( self.ecc_anom[-1] ) ]
        # Sat distance - R
        self.sat_dist = [ self.focal_parameter / ( 1 + self.eccentricity * np.cos( self.orbit_angle[-1 ] ) ) ]
        # Satellite position (PQ-base)
        self.sat_position_pq = [ [self.sat_dist[ -1 ] * np.cos( self.orbit_angle[ -1 ] ), self.sat_dist[ -1 ] * np.sin( self.orbit_angle[ -1 ] ), 0.0] ]
        # Satellite position (XYZ-base)
        rot_matrix = self._compute_rotation_matrix()
        self.sat_position_xyz = [ np.matmul( rot_matrix, np.array( self.sat_position_pq[ -1 ] ).T ) ]

        # Satellite vector basis
        self._sat_vec_basis = self._get_ref_vectorial_base()

        # Beam direction vector (Earth and Satellite VB)
        self._beam_vector_be = self._get_beam_vector_be()
        self._beam_vector_bs = self._get_beam_vector_bs()
        self._beam_dir_el_az = self._get_el_az_angles_beam()

        # Satellite antenna
        self.antenna = SatAntenna( self._beam_vector_be, ant_hpbw )
        self.tx_power = tx_power

        self.coverage_data = {}

    def _get_ref_vectorial_base( self ):
        """ Determines the satellite's reference vector basis and the basis matrix.
        Returns:
            vec_basis (VecBasis): Vector basis.
        """
        # Compute basis vectors
        r_u = ge.Vector( self.sat_position_xyz[ -1 ] ).norm()
        v_u = ge.Vector( [ -r_u.dy, r_u.dx, 0.0 ] ).norm()
        t_u = r_u.cross( v_u ).norm()

        # Get vector basis
        vec_basis = ge.VectorBasis( v_u, r_u, t_u )
        return vec_basis

    def _get_beam_vector_be( self ):
        """ Calculates the beam direction vector referenced in the terrestrial vector basis.
        Returns:
            beam_vector_be (Vector): beam direction vector (Earth vector basis)
        """
        # Target point of earth surface
        earth_surface_pos = [0,0,0]
        # Beam vector on earth basis
        beam_vector_be = (ge.Vector( earth_surface_pos ) - ge.Vector( self.sat_position_xyz[-1] )).norm()

        return beam_vector_be

    def _get_beam_vector_bs( self ):
        """ Calculates the beam direction vector referenced in the satellite vector basis.
        Returns:

        """
        # Beam vector on satellite basis
        beam_vector_base_s = np.linalg.inv( self._sat_vec_basis.basis_mat ).dot( self._beam_vector_be.asarray() )
        beam_vector_bs = ge.Vector( beam_vector_base_s ).norm()
        return beam_vector_bs

    def _get_el_az_angles_beam( self ):
        """Calculates the azimuth and elevation angles of the beam direction with 
        respect to the satellite base vector.
        """
        # Azimuth and elevation angles referenced to the satellite basis
        az = np.arctan2( self._beam_vector_bs.dy, self._beam_vector_bs.dx )
        el = np.arccos( self._beam_vector_bs.dz )

        return el, az

    def _update_beam_vector( self ):
        """Updates the beam direction vector referenced in the earth vector basis.
        """
        # Azimuth and elevation angles on the reference basis
        az = self._beam_dir_el_az[1]
        el = self._beam_dir_el_az[0]
        # Update the beam vector 
        cart_p = sph2cart(1.0, el, az)

        self._beam_vector_be = cart_p[ 0 ] * self._sat_vec_basis.v1 + \
            cart_p[ 1 ] * self._sat_vec_basis.v2 + \
            cart_p[ 2 ] * self._sat_vec_basis.v3

        self.antenna.beam_direction_vector = self._beam_vector_be
        self.antenna.vec_basis = self.antenna.get_antenna_vec_basis( self._beam_vector_be.asarray() )

        return

    def _compute_line_of_nodes( self ):

        n_0 = np.sqrt( PhyConstants.GRAV_CONSTANT.value / ((self.semi_major_axis)**3) )
        p1 = 1.5 * (PhyConstants.EARTH_NON_SPH.value * (PhyConstants.EARTH_RAD_KM.value)**2) / (self.focal_parameter**2)
        p2 = (1 - 1.5 * (np.sin( self.inclination_angle )**2)) * np.sqrt( 1.0 - (self.eccentricity**2) )
        line_of_nodes = n_0 * ( 1 + p1 * p2 )

        return line_of_nodes

    def _compute_rate_asc( self ):

        p1 = -1.5 * (PhyConstants.EARTH_NON_SPH.value * (PhyConstants.EARTH_RAD_KM.value)**2) / (self.focal_parameter**2)
        rate_asc = p1 * self.line_of_nodes * np.cos( self.inclination_angle )

        return rate_asc

    def _compute_perigee_arg_prec( self ):

        p1 = 1.5 * (PhyConstants.EARTH_NON_SPH.value * (PhyConstants.EARTH_RAD_KM.value)**2) / (self.focal_parameter**2)
        perigee_arg_prec = p1 * self.line_of_nodes * ( 2.0 - 2.5 * (np.sin( self.inclination_angle )**2) )

        return perigee_arg_prec

    def _compute_init_ecc_anomaly( self ):

        e1 = np.sqrt( ( 1.0 + self.eccentricity ) / ( 1.0 - self.eccentricity ) )
        p1 = np.tan( self.orbit_angle[ -1 ] / 2.0 ) / e1
        ecc_anomaly = 2.0 * np.arctan( p1 )

        return ecc_anomaly

    def _solve_ecc_anomaly( self ):

        # Solve Kepler equation by Newton-Raphson method
        func = lambda x : x - self.eccentricity * np.sin(x) - self.mean_anom[-1]
        # x = symbols('x')
        # expr = x - self.eccentricity * sin(x) - self.current_mean_anom
        sol = fsolve(func, self.ecc_anom[-1])

        return sol[0]

    def _compute_rotation_matrix( self ):

        omega = self.asc_node_long[ -1 ]
        omega_m = self.perigee_arg[ -1 ]
        i = self.inclination_angle
        # Rotation matrix
        r_11 = np.cos( omega ) * np.cos( omega_m ) - np.sin( omega ) * np.sin( omega_m ) * np.cos( i )
        r_12 = -np.cos( omega ) * np.sin( omega_m ) - np.sin( omega ) * np.cos( omega_m ) * np.cos( i )
        r_13 = np.sin( omega ) * np.sin( i )
        r_21 = np.sin( omega ) * np.cos( omega_m ) + np.cos( omega ) * np.sin( omega_m ) * np.cos( i )
        r_22 = -np.sin( omega ) * np.sin( omega_m ) + np.cos( omega ) * np.cos( omega_m ) * np.cos( i )
        r_23 = -np.cos( omega ) * np.sin( i )
        r_31 = np.sin( omega_m ) * np.sin( i )
        r_32 = np.cos( omega_m ) * np.sin( i )
        r_33 = np.cos( i )
        rot_mat = np.array( [[ r_11, r_12, r_13 ], [ r_21, r_22, r_23 ], [ r_31, r_32, r_33 ]] )

        return rot_mat

    def update_position( self, dt: float ):

        # Add the new time instant to the list
        self.time.append( self.time[ -1 ] + dt )

        # Perigee argument
        self.perigee_arg.append( self.perigee_arg[ 0 ] + self.perigee_arg_prec * self.time[ -1 ] )
        # Ascending node longitude
        self.asc_node_long.append( self.asc_node_long[ 0 ] + self.rate_asc * self.time[ -1 ] )
        # Mean anomaly
        self.mean_anom.append( self.mean_anom[ 0 ] + self.line_of_nodes * self.time[ -1 ] )
        # Ecc anomaly
        self.ecc_anom.append( self._solve_ecc_anomaly() )
        # True anomaly
        e1 = np.sqrt( ( 1.0 + self.eccentricity ) / ( 1.0 - self.eccentricity ) ) * np.tan( self.ecc_anom[ -1 ] / 2.0 )
        self.orbit_angle.append( 2.0 * np.arctan( e1 ) )

        # Satellite distance
        self.sat_dist.append( self.focal_parameter / ( 1 + self.eccentricity * np.cos( self.orbit_angle[ -1 ] ) ) )
        # Satellite position P-Q
        self.sat_position_pq.append( [ self.sat_dist[ -1 ] * np.cos( self.orbit_angle[ -1 ] ), self.sat_dist[ -1 ] * np.sin( self.orbit_angle[ -1 ] ), 0.0 ] )
        # Satellite position xyz
        rot_matrix = self._compute_rotation_matrix()
        self.sat_position_xyz.append( list( np.matmul( rot_matrix, np.array( self.sat_position_pq[ -1 ] ).T ) ) )

        # Update vector basis
        self._sat_vec_basis = self._get_ref_vectorial_base()
        # Update beam vector
        self._update_beam_vector()

        return

    def plot_sat( self, return_fig = False, vecs_exag = 10000 ):
        """ Plots satellite position, base vector, pointing vector and terrestrial sphere.
        Returns:
            Object of the plotted figure.
        """

        fig = plt.figure(figsize=(10,10), dpi=80)
        
        ax = fig.add_subplot(111, projection='3d')
        # Make data
        r = PhyConstants.EARTH_RAD_KM.value
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        ax.plot_surface(x, y, z, color='linen', alpha=0.5)
        # plot circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        ax.plot(x, y, z, color='black', alpha=0.75)
        ax.plot(z, x, y, color='black', alpha=0.75)

        ## add axis lines
        zeros = np.zeros(1000)
        line = np.linspace(-r,r,1000)

        ax.plot(line, zeros, zeros, color='black', alpha=0.75)
        ax.plot(zeros, line, zeros, color='black', alpha=0.75)
        ax.plot(zeros, zeros, line, color='black', alpha=0.75)

        # Plot Ngeo satellite
        scat = ax.scatter( *(self.sat_position_xyz[-1]), marker='x', linewidth=4, s=150, color='blue' )
        # Plot vector basis
        r = vecs_exag
        ax.quiver( *(self.sat_position_xyz[-1]), r * self._sat_vec_basis.v1.dx, r * self._sat_vec_basis.v1.dy, r * self._sat_vec_basis.v1.dz, color='blue')
        ax.quiver( *(self.sat_position_xyz[-1]), r * self._sat_vec_basis.v2.dx, r * self._sat_vec_basis.v2.dy, r * self._sat_vec_basis.v2.dz, color='red')
        ax.quiver( *(self.sat_position_xyz[-1]), r * self._sat_vec_basis.v3.dx, r * self._sat_vec_basis.v3.dy, r * self._sat_vec_basis.v3.dz, color='black')
        ax.quiver( *(self.sat_position_xyz[-1]), r * self._beam_vector_be.dx, r * self._beam_vector_be.dy, r * self._beam_vector_be.dz, color='green')

        ax.set_zlim(-2.0 * (self.apogee_dist + PhyConstants.EARTH_RAD_KM.value) , 2.0 * (self.apogee_dist + PhyConstants.EARTH_RAD_KM.value))
        ax.set_ylim(-2.0 * (self.apogee_dist + PhyConstants.EARTH_RAD_KM.value) , 2.0 * (self.apogee_dist + PhyConstants.EARTH_RAD_KM.value))
        ax.set_xlim(-2.0 * (self.apogee_dist + PhyConstants.EARTH_RAD_KM.value) , 2.0 * (self.apogee_dist + PhyConstants.EARTH_RAD_KM.value))

        # ret = fig if return_fig else None
        
        ret = None
        if return_fig:
            ret = fig
        else:
            plt.show()

        return ret



class System:

    # Angular resolution of rays in the elevation domain
    cone_el_angle_resol = np.pi / 1000.0
    # Maximum area threshold
    area_th = 360 * 180 / 4
    # Grid resolution
    grid_resol = 500
    # Space thresholds
    x_th = 358
    y_th = 178

    def __init__(self, 
                 earth: Earth, 
                 earth_stations_net: List[EarthStation], 
                 geo_sat_net: List[GeoSatellite], 
                 ngeo_sat_net: List[NGeoSatellite],
                 frequency: float,
                 cone_aperture: float = 1.0 ) -> None:

        self.earth = earth # Earth reference
        self.earth_station_net = earth_stations_net 
        self.geo_sat_net = { geo_sat.name: geo_sat for geo_sat in geo_sat_net } # Geo satellites list
        self.ngeo_sat_net = { ngeo_sat.name: ngeo_sat for ngeo_sat in ngeo_sat_net } # NGeo satellites list
        self.time = [0]
        self.st_id = 0
        self.cone_aperture = cone_aperture
        self.frequency = frequency
        self.wavelength = PhyConstants.LIGHT_SPEED.value / self.frequency
        self.coverage_zones = [self.compute_coverage_zones()]
        
    def compute_coverage_zones( self ):
        """ Calculates the areas covered by satellites.
        Returns:
            dict: Dictionary with the relationships between the satellite 
            and the demarcation of coverage areas.
        """

        # Compensate for Earth's rotation
        dg = PhyConstants.EARTH_ANG_RATE_ROT_DG.value * self.time[ -1 ]
        dr = PhyConstants.EARTH_ANG_RATE_ROT_RAD.value * self.time[ -1 ]
        # Earth rotation matrix
        rot_matrix = self.earth._get_rotation_mat( dr )
        rot_matrix_m = self.earth._get_rotation_mat( -dr )

        # Interaction on Geo satellites
        for geo_sat_name, geo_sat in self.geo_sat_net.items():

            aux_list = []
            # Ray origin
            ray_origin = ge.Point( *geo_sat.current_cart )

            # El-Az angles of the cone
            max_cone_aperture = self.cone_aperture * geo_sat.antenna.hpbw_rad
            azv = np.linspace( 0, 2 * np.pi, 100 )
            elv = np.linspace( max_cone_aperture, 0, 1000 )
            for az in azv:
                for el in elv:
                    
                    cart_p = sph2cart(1.0, el, az)
                    ray_dir_vector = cart_p[ 0 ] * geo_sat.antenna.vec_basis.v1 + \
                        cart_p[ 1 ] * geo_sat.antenna.vec_basis.v2 + \
                        cart_p[ 2 ] * geo_sat.antenna.vec_basis.v3
                    # Ray
                    ray = ge.Ray( ray_origin, ray_dir_vector )
                    # Intersection point
                    intersection_point = self.earth.sphere.intersect( ray )
                    if intersection_point is not None:
                        aux_list.append( np.matmul( rot_matrix_m, intersection_point.asarray() ) )
                        break

            geo_sat.coverage_data[ self.st_id ] = {}
            geo_sat.coverage_data[ self.st_id ][ 'Array' ] = np.array( aux_list )
            geo_sat.coverage_data[ self.st_id ][ 'LatLong' ] = sort_coordinates( np.rad2deg( cart_2_ll( np.array( aux_list ) ) ) )

            # Corrige deslocamento
            fx = False
            fy = False
            cov = geo_sat.coverage_data[ self.st_id ][ 'LatLong' ]
            x_range = np.max( cov[ :, 0 ] ) - np.min( cov[ :, 0 ] )
            y_range = np.max( cov[ :, 1 ] ) - np.min( cov[ :, 1 ] )
            if np.any( x_range >= self.x_th  ):
                fx = True
                geo_sat.coverage_data[ self.st_id ][ 'LatLong' ][ cov[ :, 0 ] < 0.0, 0 ] = cov[ cov[ :, 0 ] < 0.0, 0 ] + 360
            if np.any( y_range >= self.y_th ):
                fy = False
                geo_sat.coverage_data[ self.st_id ][ 'LatLong' ][ cov[ :, 1 ] < 0.0, 1 ] = cov[ cov[ :, 1 ] < 0.0, 1 ] + 180
            geo_sat.coverage_data[ self.st_id ][ 'LatLong' ] = sort_coordinates( geo_sat.coverage_data[ self.st_id ][ 'LatLong' ] )

            # Get polygon
            poly = Polygon( geo_sat.coverage_data[ self.st_id ][ 'LatLong' ] )
            geo_sat.coverage_data[ self.st_id ][ 'Poly' ] = poly

            # Bounding box
            minx_p, miny_p, maxx_p, maxy_p = poly.bounds
            minx, miny, maxx, maxy = -180, -90, 180, 90
            geo_sat.coverage_data[ self.st_id ][ 'Cov' ] = np.nan * np.ones(shape=( self.grid_resol, self.grid_resol ))
            geo_sat.coverage_data[ self.st_id ][ 'PFD' ] = np.nan * np.ones(shape=( self.grid_resol, self.grid_resol ))
            # Long
            xv = np.linspace( minx, maxx, self.grid_resol )
            # Lat
            yv = np.linspace( miny, maxy, self.grid_resol )
            # Compute coverages
            for i, x in enumerate(xv):
                for j, y in enumerate(yv):

                    # Shift in case of polygon rotation
                    xp = x + 360 if ( x < 0.0 and fx is True ) else x
                    yp = y + 180 if ( y < 0.0 and fy is True ) else y

                    if ( minx_p <= xp <= maxx_p ) and ( miny_p <= yp <= maxy_p ):
                        if poly.contains( Point( xp, yp ) ):

                            # Convert to spherical
                            az, el = latlong2azel( np.deg2rad( y ), np.deg2rad( x ) )
                            es_point = ge.Point( *np.matmul( rot_matrix, np.array( sph2cart( self.earth.earth_rad, el, az) ) ) )

                            # Compute the distance
                            sat_to_point_vec = es_point - ray_origin
                            sat_to_point_dist = ge.length( sat_to_point_vec ) * 1000

                            # Antenna Gains
                            tx_ant_gain = geo_sat.antenna.ant_gain_on_direction( sat_to_point_vec )
                            geo_sat.coverage_data[ self.st_id ][ 'Cov' ][ j, i ] = \
                                10.0 * np.log10( tx_ant_gain * geo_sat.tx_power * ( self.wavelength / ( 4.0 * np.pi * sat_to_point_dist ) )**2 )
                            geo_sat.coverage_data[ self.st_id ][ 'PFD' ][ j, i ] = \
                                10.0 * np.log10( tx_ant_gain * geo_sat.tx_power * ( 1.0 / ( 4.0 * np.pi * sat_to_point_dist**2 ) ) )

        # Interaction on NGeo satellites
        for ngeo_sat_name, ngeo_sat in self.ngeo_sat_net.items():

            aux_list = []
            # Ray origin
            ray_origin = ge.Point( *ngeo_sat.sat_position_xyz[ -1 ] )

            # El-Az angles of the cone
            max_cone_aperture = self.cone_aperture * ngeo_sat.antenna.hpbw_rad
            azv = np.linspace( 0, 2 * np.pi, 100 )
            elv = np.linspace( max_cone_aperture, 0, 1000 )
            for az in azv:
                for el in elv:
                    
                    cart_p = sph2cart(1.0, el, az)
                    ray_dir_vector = cart_p[ 0 ] * ngeo_sat.antenna.vec_basis.v1 + \
                        cart_p[ 1 ] * ngeo_sat.antenna.vec_basis.v2 + \
                        cart_p[ 2 ] * ngeo_sat.antenna.vec_basis.v3
                    # Ray
                    ray = ge.Ray( ray_origin, ray_dir_vector )
                    # Intersection point
                    intersection_point = self.earth.sphere.intersect( ray )
                    if intersection_point is not None:
                        aux_list.append( np.matmul( rot_matrix_m, intersection_point.asarray() ) )
                        break

            ngeo_sat.coverage_data[ self.st_id ] = {}
            ngeo_sat.coverage_data[ self.st_id ][ 'Array' ] = np.array( aux_list )
            ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ] = np.rad2deg( cart_2_ll( np.array( aux_list ) ) )

            # Corrige deslocamento
            fx = False
            fy = False
            cov = ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ]
            x_range = np.max( cov[ :, 0 ] ) - np.min( cov[ :, 0 ] )
            y_range = np.max( cov[ :, 1 ] ) - np.min( cov[ :, 1 ] )
            if np.any( x_range >= self.x_th  ):
                fx = True
                ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ][ cov[ :, 0 ] < 0.0, 0 ] = cov[ cov[ :, 0 ] < 0.0, 0 ] + 360
            if np.any( y_range >= self.y_th ):
                fy = False
                ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ][ cov[ :, 1 ] < 0.0, 1 ] = cov[ cov[ :, 1 ] < 0.0, 1 ] + 180
            ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ] = sort_coordinates( ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ] )

            # Get polygon
            poly = Polygon( ngeo_sat.coverage_data[ self.st_id ][ 'LatLong' ] )
            ngeo_sat.coverage_data[ self.st_id ][ 'Poly' ] = poly

            # Bounding box
            minx_p, miny_p, maxx_p, maxy_p = poly.bounds
            minx, miny, maxx, maxy = -180, -90, 180, 90
            ngeo_sat.coverage_data[ self.st_id ][ 'Cov' ] = np.nan * np.ones(shape=( self.grid_resol, self.grid_resol ))
            ngeo_sat.coverage_data[ self.st_id ][ 'PFD' ] = np.nan * np.ones(shape=( self.grid_resol, self.grid_resol ))
            # Long
            xv = np.linspace( minx, maxx, self.grid_resol )
            # Lat
            yv = np.linspace( miny, maxy, self.grid_resol )
            # Compute coverages
            for i, x in enumerate(xv):
                for j, y in enumerate(yv):

                    # Shift in case of polygon rotation
                    xp = x + 360 if ( x < 0.0 and fx is True ) else x
                    yp = y + 180 if ( y < 0.0 and fy is True ) else y

                    if ( minx_p <= xp <= maxx_p ) and ( miny_p <= yp <= maxy_p ):
                        if poly.contains( Point( xp, yp ) ):

                            # Convert to spherical
                            az, el = latlong2azel( np.deg2rad( y ), np.deg2rad( x ) )
                            es_point = ge.Point( *np.matmul( rot_matrix, np.array( sph2cart( self.earth.earth_rad, el, az) ) ) )
                            # Compute the distance
                            sat_to_point_vec = es_point - ray_origin
                            sat_to_point_dist = ge.length( sat_to_point_vec ) * 1000

                            # Antenna Gains
                            tx_ant_gain = ngeo_sat.antenna.ant_gain_on_direction( sat_to_point_vec )
                            ngeo_sat.coverage_data[ self.st_id ][ 'Cov' ][ j, i ] = \
                                10.0 * np.log10( tx_ant_gain * ngeo_sat.tx_power * ( self.wavelength / ( 4.0 * np.pi * sat_to_point_dist ) )**2 )
                            ngeo_sat.coverage_data[ self.st_id ][ 'PFD' ][ j, i ] = \
                                10.0 * np.log10( tx_ant_gain * ngeo_sat.tx_power * ( 1.0 / ( 4.0 * np.pi * sat_to_point_dist**2 ) ) )

 
        return

    # def compute_interference_zones( self ):

    #     # Interaction on Geo satellites
    #     for geo_sat_name, geo_sat in self.geo_sat_net.items():

    #         # Search for interfering satellites
    #         for int_sat_name, int_sat in self.geo_sat_net.items():

    #             # Jump the main satellite
    #             if int_sat_name == geo_sat_name:
    #                 continue
    #             # Main sat coverage poly
    #             m_poly = geo_sat.coverage_data[ self.st_id ][ 'Poly' ]
    #             # Interfering coverage poly
    #             i_poly = int_sat.coverage_data[ self.st_id ][ 'Poly' ]

    #             int_poly = m_poly.boundary.intersection(i_poly)

    #             print(int_poly.convex_hull)
    #             print(m_poly)

    #             return int_poly.convex_hull


    def plot_current_coverage_zone( self, sat_name: str ):

        # Plot earth basemap
        fig = plt.figure(figsize=(12, 10), edgecolor='w')
        ax = plt.gca()
        basemap = Basemap(projection='cyl', resolution='c', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180 )
        # draw a shaded-relief image
        scale=0.2
        basemap.shadedrelief(scale=scale)
        # lats and longs are returned as a dictionary
        lats = basemap.drawparallels(np.linspace(-90, 90, 13), labels=[1,1,0,1], fontsize=8)
        lons = basemap.drawmeridians(np.linspace(-180, 180, 13), labels=[1,1,0,1], fontsize=8)
        # keys contain the plt.Line2D instances
        lat_lines = chain(*(tup[1][0] for tup in lats.items()))
        lon_lines = chain(*(tup[1][0] for tup in lons.items()))
        all_lines = chain(lat_lines, lon_lines)
        # cycle through these lines and set the desired style
        for line in all_lines:
            line.set(linestyle='-', alpha=0.3, color='w')
        plt.xlabel('Longitude', labelpad=40, fontsize=8)
        plt.ylabel('Latitude', labelpad=40, fontsize=8)

        # Get the last coverage zones map
        sat = None
        if sat_name in self.geo_sat_net:
            sat = self.geo_sat_net[ sat_name ]
        elif sat_name in self.ngeo_sat_net:
            sat = self.ngeo_sat_net[ sat_name ]
        current_st_id = self.st_id
        if sat:

            # Title
            plt.title( f'Zona de cobertura - {sat_name}, Abertura: {self.cone_aperture} x { sat.antenna.hpbw_dg }°' )
            # Coverage zone dict
            cov_dict = sat.coverage_data[ current_st_id ]

            # Contour mask
            c_mask = np.zeros( cov_dict['Cov'].shape, dtype=bool )
            # False para os valores abaixo do limiar
            c_mask[ ~np.isnan( cov_dict['Cov'] ) ] = True
            c_mask[ np.isnan( cov_dict['Cov'] ) ] = False
            
            # Coverage bounding box
            cov_bbox = -180, 180, -90, 90
            # Plotar contorno
            plt.contour(c_mask, [0.5], colors="red", linewidths=1, extent=cov_bbox, linestyles='dashed' )
            # plt.legend()
            # Mapa de calor
            im = plt.imshow( cov_dict['Cov'], extent=cov_bbox, origin='lower' )
            # Barra de cores
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.5)
            plt.colorbar(im, cax=cax)

            # for ip in range(0, len( cov_dict[ 'Poly' ] ) ):

            #     # Coverage bounding box
            #     cov_bbox = cov_dict[ 'BBox' ][ ip ]
            #     # Coverage zone polygon
            #     xs, ys = cov_dict[ 'Poly' ][ ip ].exterior.xy 
            #     # Plot coverage zone contour
            #     basemap.plot( cov_dict[ 'LatLong' ][ :, 0 ], cov_dict[ 'LatLong' ][ :, 1 ], latlon=True, linestyle='--', label=sat_name, color='red', linewidth=1.5 )
            #     # Legenda
            #     plt.legend()
            #     # Mapa de calor
            #     im = plt.imshow( cov_dict['Cov'][ip], extent=cov_bbox, origin='lower' )
            #     # basemap.plot( np.rad2deg( self.geo_sat_net[ sat_name ]._init_lat_lon_target_position[1] ), np.rad2deg( self.geo_sat_net[ sat_name ]._init_lat_lon_target_position[0] ), latlon=True, marker='x', label='Centro do Beam', linestyle='None', color='red', linewidth=1.5 )

            #     # Barra de cores
            #     divider = make_axes_locatable(ax)
            #     cax = divider.append_axes("right", size="2%", pad=0.5)
            #     plt.colorbar(im, cax=cax)




        plt.show()

        return

    def update_system( self, dt: float ):

        for es in self.earth_station_net:
            es.update_position(dt)

        for geo_s_name, geo_s in self.geo_sat_net.items():
            geo_s.update_position(dt)
        
        for ngeo_s_name, ngeo_s in self.ngeo_sat_net.items():
            ngeo_s.update_position(dt)

        # Increment time and state
        self.time.append( self.time[ -1 ] + dt )
        self.st_id = self.st_id + 1
        # Compute coverage zones
        self.compute_coverage_zones()

        return

    def plot_system( self, vecs_exag=10000, return_fig=False ):
        """ Plot the earth station coordinates """

        fig = plt.figure(figsize=(10,10), dpi=80)
        
        ax = fig.add_subplot(111, projection='3d')
        # Make data
        r = PhyConstants.EARTH_RAD_KM.value
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        ax.plot_surface(x, y, z, color='linen', alpha=0.5)
        # plot circular curves over the surface
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.zeros(100)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        ax.plot(x, y, z, color='black', alpha=0.75)
        ax.plot(z, x, y, color='black', alpha=0.75)

        ## add axis lines
        zeros = np.zeros(1000)
        line = np.linspace(-r,r,1000)

        ax.plot(line, zeros, zeros, color='black', alpha=0.75)
        ax.plot(zeros, line, zeros, color='black', alpha=0.75)
        ax.plot(zeros, zeros, line, color='black', alpha=0.75)

        # Plot earth stations coordinates
        for es in self.earth_station_net:
            ax.scatter( *es.es_current_coords_cart, marker='+', linewidth=4, s=150, color='black' )

        # Plot geo satellites
        for geo_s_name, geo_s in self.geo_sat_net.items():
            # Plot geo satellite
            ax.scatter( *geo_s.current_cart, marker='x', linewidth=4, s=150, color='blue' )
            # Plot vector basis
            r = vecs_exag
            ax.quiver( *geo_s.current_cart, r * geo_s._sat_vec_basis.v1.dx, r * geo_s._sat_vec_basis.v1.dy, r * geo_s._sat_vec_basis.v1.dz, color='blue')
            ax.quiver( *geo_s.current_cart, r * geo_s._sat_vec_basis.v2.dx, r * geo_s._sat_vec_basis.v2.dy, r * geo_s._sat_vec_basis.v2.dz, color='red')
            ax.quiver( *geo_s.current_cart, r * geo_s._sat_vec_basis.v3.dx, r * geo_s._sat_vec_basis.v3.dy, r * geo_s._sat_vec_basis.v3.dz, color='black')
            ax.quiver( *geo_s.current_cart, r * geo_s._beam_vector_be.dx, r * geo_s._beam_vector_be.dy, r * geo_s._beam_vector_be.dz, color='green')

        # Plot geo satellites
        for ngeo_s_name, ngeo_s in self.ngeo_sat_net.items():
            # Plot Ngeo satellite
            ax.scatter( *(ngeo_s.sat_position_xyz[-1]), marker='^', linewidth=4, s=150, color='green' )
            # Plot vector basis
            r = vecs_exag / 2
            ax.quiver( *(ngeo_s.sat_position_xyz[-1]), r * ngeo_s._sat_vec_basis.v1.dx, r * ngeo_s._sat_vec_basis.v1.dy, r * ngeo_s._sat_vec_basis.v1.dz, color='blue')
            ax.quiver( *(ngeo_s.sat_position_xyz[-1]), r * ngeo_s._sat_vec_basis.v2.dx, r * ngeo_s._sat_vec_basis.v2.dy, r * ngeo_s._sat_vec_basis.v2.dz, color='red')
            ax.quiver( *(ngeo_s.sat_position_xyz[-1]), r * ngeo_s._sat_vec_basis.v3.dx, r * ngeo_s._sat_vec_basis.v3.dy, r * ngeo_s._sat_vec_basis.v3.dz, color='black')
            ax.quiver( *(ngeo_s.sat_position_xyz[-1]), r * ngeo_s._beam_vector_be.dx, r * ngeo_s._beam_vector_be.dy, r * ngeo_s._beam_vector_be.dz, color='green')


        ax.set_zlim(-PhyConstants.GEO_ORBIT_RAD_KM.value, PhyConstants.GEO_ORBIT_RAD_KM.value)
        ax.set_ylim(-PhyConstants.GEO_ORBIT_RAD_KM.value, PhyConstants.GEO_ORBIT_RAD_KM.value)
        ax.set_xlim(-PhyConstants.GEO_ORBIT_RAD_KM.value, PhyConstants.GEO_ORBIT_RAD_KM.value)
        # ax.axis('off')
        # plt.show()

        ret = None
        if return_fig:
            ret = fig
        else:
            plt.show()

        return ret

