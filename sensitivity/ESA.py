from metpy.units import units
from metpy.calc import potential_temperature, lat_lon_grid_deltas
from metpy.calc import potential_vorticity_baroclinic as pv_bc

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xarray as xr

from numba import jit, prange, njit
from joblib import Parallel, delayed

import numpy as np

upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}


def calculate_potential_vorticity(temperature, pressure, u_wind, v_wind, lat_grid, lon_grid):
    """
    Calculate potential vorticity using MetPy's baroclinic potential vorticity function.

    Parameters:
    - temperature: Temperature (NumPy array)
    - pressure: Pressure (NumPy array)
    - u_wind: Zonal wind component (NumPy array)
    - v_wind: Meridional wind component (NumPy array)
    - lat_grid: 2D Latitude grid (NumPy array, degrees)
    - lon_grid: 2D Longitude grid (NumPy array, degrees)

    Returns:
    - Potential vorticity (pint.Quantity)
    """
    # 1. Convert all inputs to pint.Quantity with correct units
    temperature_q = temperature * units.K
    pressure_q = (pressure * units.hPa).to(units.Pa) # Convert hPa to Pa
    u_wind_q = u_wind * units.m / units.s
    v_wind_q = v_wind * units.m / units.s
    lat_grid_q = lat_grid * units.degrees
    lon_grid_q = lon_grid * units.degrees

    # 2. Calculate potential temperature (theta)
    theta = potential_temperature(pressure_q, temperature_q)

    # 3. Calculate dx and dy from the 2D lat/lon grids
    dx_grid_2d, dy_grid_2d = lat_lon_grid_deltas(lon_grid_q, lat_grid_q)

    # --- REVISED: Expand dx_grid and dy_grid to 3D ---
    # Assuming the 3D data shape is (level, latitude, longitude)
    # Add a new axis at the beginning (for the level dimension)
    dx_grid = dx_grid_2d[np.newaxis, :, :]
    dy_grid = dy_grid_2d[np.newaxis, :, :]

    # To ensure broadcasting works correctly, make sure dx_grid and dy_grid
    # have the same number of levels as your input data.
    # You can broadcast them explicitly to the shape of your 3D data,
    # or MetPy's internal broadcasting might handle it if the newaxis is correct.
    # Let's explicitly broadcast for clarity and robustness:
    num_levels = temperature_q.shape[0] # Assuming level is the first dimension
    dx_grid = np.broadcast_to(dx_grid, (temperature_q.shape[0], *dx_grid_2d.shape))
    dy_grid = np.broadcast_to(dy_grid, (temperature_q.shape[0], *dy_grid_2d.shape))
    # ---------------------------------------------------

    # 4. Calculate potential vorticity
    pv = pv_bc(
        potential_temperature=theta,
        pressure=pressure_q,
        u=u_wind_q,
        v=v_wind_q,
        dx=dx_grid, # Use the 3D dx_grid
        dy=dy_grid, # Use the 3D dy_grid
        latitude=lat_grid_q, # Latitude is used for Coriolis, can remain 2D or 3D
        # --- ADDED: Explicitly define the dimension indices ---
        # Assuming your 3D data is (level, latitude, longitude)
        vertical_dim=0, # Index for level dimension
        y_dim=1,        # Index for latitude dimension
        x_dim=2         # Index for longitude dimension
        # ---------------------------------------------------
    )

    return pv


# Bring data from ensemble members
def process_ens(input_data_dir: str, ens: Union[str, int], predict_interval: str,
                pres_array: np.ndarray = pres_array, lon_grid: np.ndarray = None, lat_grid: np.ndarray = None,
                level: Union[int, float, list, np.ndarray] = 850) -> Tuple[np.ndarray, np.ndarray]:
    upper = np.load(os.path.join(
        f'{input_data_dir}/{ens}/upper/{predict_interval}h.npy',
    )).astype(np.float32)
    # z, q 값 단위 변환
    upper[upper_dict['z']] /= 9.80665
    upper[upper_dict['q']] *= 1000

    # 레벨별 변수 추출
    if isinstance(level, (int, float)):
        level = [level]
    level = np.array(level, dtype=np.float32)
    level_idx = [np.where(pres_array == lv)[0][0] for lv in level]
    upper_part_level = upper[:, level_idx, :, :]  # Extract variables at requested levels


    # q 적분
    q_all = upper[upper_dict['q'], :]
    q_col = np.trapz(q_all[::-1], x=pres_array[::-1], axis=0)/9.80665/10  # 압력방향 뒤집기(보통 [높은압력, ..., 낮은압력]으로 적분)


    pv = calculate_potential_vorticity(
    temperature=upper[upper_dict['t']],
    pressure=np.broadcast_to(pres_array[:, np.newaxis, np.newaxis], upper[upper_dict['t']].shape),
    u_wind=upper[upper_dict['u']],
    v_wind=upper[upper_dict['v']],
    lat_grid=lat_grid,
    lon_grid=lon_grid
    )

    return upper_part_level, q_col, pv


# Compute variance vector for flattened distances
@njit(parallel=True)
def var_axis0(arr):
    N, M = arr.shape
    out = np.zeros(M)
    for j in prange(M):
        mean = np.mean(arr[:, j])
        acc = 0.0
        for i in range(N):
            acc += (arr[i, j] - mean) ** 2
        out[j] = acc / (N - 1)  # ddof=1
    return out

# Compute mean vector for flattened distances
@njit(parallel=True)
def mean_axis0(arr):
    N, M = arr.shape
    out = np.zeros(M)
    for j in prange(M):
        acc = 0.0
        for i in range(N):
            acc += arr[i, j]
        out[j] = acc / N
    return out

# Compute covariance vector for flattened distances and S matrix
@njit(parallel=True)
def compute_cov_vector(distances: np.ndarray, S: np.ndarray) -> np.ndarray:  # distances: (N,), S: (N, M)
    N, M = S.shape
    cov_vector = np.zeros(M)
    mean_d = 0.0
    for i in range(N):
        mean_d += distances[i]
    mean_d /= N

    for j in prange(M):  # 여기만 prange!
        mean_s = 0.0
        for i in range(N):
            mean_s += S[i, j]
        mean_s /= N

        cov = 0.0
        for i in range(N):
            cov += (distances[i] - mean_d) * (S[i, j] - mean_s)
        cov_vector[j] = cov / (N - 1)
    
    return cov_vector

#! Compute covariance/variance matrix for the given data and distances
def compute_cov_var(x_array: np.ndarray, distances_tar: np.ndarray, ens_idx_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cov/var matrix for the given data and distances.
    
    Parameters:
    - x_array: Data array of shape (N, lat, lon)
    - distances_tar: Distances array of shape (N,)
    - ens_idx_list: List of ensemble indices
    
    Returns:
    - cov_var: Covariance/variance matrix of shape (lat, lon)
    """
    # Normalize x_array
    var_x_array = var_axis0(x_array.reshape(len(ens_idx_list), -1))
    var_x_array = var_x_array.reshape(x_array.shape[1], x_array.shape[2])
    mean_x_array = mean_axis0(x_array.reshape(len(ens_idx_list), -1))
    mean_x_array = mean_x_array.reshape(x_array.shape[1], x_array.shape[2])
    x_array = (x_array - mean_x_array)/np.sqrt(var_x_array)

    # Compute covariance vector, variance of x_array == 1
    cov_vector = compute_cov_vector(distances_tar, x_array.reshape(len(ens_idx_list), -1))
    cov_matrix = cov_vector.reshape(x_array.shape[1], x_array.shape[2])
    cov_var = np.array(cov_matrix)
    return cov_var

def map_gridlines(ax, proj = ccrs.PlateCarree(), font_size = 15, gridline_spaces = 10, zorder = 0):
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder= zorder)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': font_size}
    gl.ylabel_style = {'size': font_size}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, gridline_spaces))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, gridline_spaces))
