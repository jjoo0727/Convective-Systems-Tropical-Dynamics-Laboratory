import os
import numpy as np
import pandas as pd
import time
from math import radians, degrees, sin, cos, asin, acos, sqrt, atan2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import plotly.figure_factory as ff
import matplotlib.collections as mcoll
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from geopy.distance import geodesic
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import tcmarkers

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.decomposition import PCA
from skimage.measure import regionprops

import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, minimum_filter, maximum_filter, label
from scipy import integrate
from scipy.ndimage import gaussian_filter1d

from datetime import datetime, timedelta

from haversine import haversine

import tropycal.tracks as tracks

from numba import jit

import itertools    



pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'



pres_list = ['1000','925','850','700','600','500','400','300','250','200','150','100','50']
pres=500                                                #살펴볼 기압면 결정
p=pres_list.index(str(pres))
pres_array = np.array(pres_list, dtype=np.float32)

surface_factor = ['MSLP', 'U10', 'V10', 'T2M']
surface_dict = {'MSLP':0, 'U10':1, 'V10':2, 'T2M':3}
upper_factor = ['z', 'q', 't', 'u', 'v']
upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}

proj = ccrs.PlateCarree()
norm_p = mcolors.Normalize(vmin=950, vmax=1020)

#위경도 범위 지정 함수
def latlon_extent(lon_min, lon_max, lat_min, lat_max, part = 'y'):    
    # lon_min, lon_max = lon_min, lon_max  
    if part == 'y':
        lat_indices = np.arange(lat_min, lat_max+0.1, 0.25)[::-1]
        lon_indices = np.arange(lon_min, lon_max+0.1, 0.25)
    else:
        lat_indices = np.linspace(90, -90, 721)
        lon_indices = np.concatenate((np.linspace(0, 180, 721), np.linspace(-180, 0, 721)[1:-1]),axis=0)
    # 위경도 범위를 데이터의 행과 열 인덱스로 변환
    lat_start = np.argmin(np.abs(lat_indices - lat_max)) 
    lat_end = np.argmin(np.abs(lat_indices - lat_min))
    lon_start = np.argmin(np.abs(lon_indices - lon_min))
    lon_end = np.argmin(np.abs(lon_indices - lon_max))
    latlon_ratio = (lon_max-lon_min)/(lat_max-lat_min)
    extent=[lon_min, lon_max, lat_min, lat_max]
    return lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio



def storm_info(pangu_dir, storm_name, storm_year, datetime_list=None, wind_thres=35):
    file_path = f'{pangu_dir}/storm_info/{storm_year}_{storm_name}.csv'
    if not os.path.exists(file_path):
        ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc',catarina=True)
        storm = ibtracs.get_storm((storm_name,storm_year))
        storm = storm.to_dataframe()
        storm.to_csv(file_path, index=False)
    
    storm = pd.read_csv(file_path)
    storm_time = np.array([datetime.strptime(i,"%Y-%m-%d %H:%M:%S") for i in storm['time']])
    
    
    if datetime_list is not None and not isinstance(datetime_list, int):
        mask = (storm['vmax'] >= wind_thres) & np.isin(storm_time, datetime_list)
    else:
        mask = storm['vmax'] >= wind_thres

    storm_lon = storm['lon'][mask].to_numpy()
    storm_lat = storm['lat'][mask].to_numpy()
    storm_mslp = storm['mslp'][mask].to_numpy()   
    storm_time = storm_time[mask]   
    
    return storm_lon, storm_lat, storm_mslp, storm_time

# @jit(nopython=True)
# def haversine_distance(lat1, lon1, lat2, lon2):
#     """
#     Calculate the great circle distance in kilometers between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#     c = 2 * np.arcsin(np.sqrt(a)) 
#     r = 6371 # Radius of earth in kilometers. Use 3956 for miles
#     return c * r

@jit(nopython=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees) using Taylor expansion for sine and cosine calculations
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula approximation with Taylor expansions
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    sin_dlat2 = (dlat/2) - (dlat**3)/48  # Taylor expansion for sin(dlat/2)
    sin_dlon2 = (dlon/2) - (dlon**3)/48  # Taylor expansion for sin(dlon/2)
    cos_lat1 = 1 - (lat1**2)/2 + (lat1**4)/24  # Taylor expansion for cos(lat1)
    cos_lat2 = 1 - (lat2**2)/2 + (lat2**4)/24  # Taylor expansion for cos(lat2)
    a = sin_dlat2**2 + cos_lat1 * cos_lat2 * sin_dlon2**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

#output 기상 정보 클래스
class Met:
    def __init__(self, output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid, input_sign = 'n'):
        if input_sign =='n':
            self.surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
            self.upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
        else:
            self.surface = np.load(os.path.join(output_data_dir, rf'surface.npy')).astype(np.float32)
            self.upper = np.load(os.path.join(output_data_dir, rf'upper.npy')).astype(np.float32)
            
        self.surface_dict = surface_dict
        self.upper_dict = upper_dict
        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lat_grid = lat_grid
        self.lon_start = lon_start
        self.lon_end = lon_end
        self.lon_grid = lon_grid
    
    @staticmethod
    def data_unit(data, name):
        if name == 'MSLP':
            data /= 100
        elif name == 'z':
            data /= 9.80665
        elif name == 'q':
            data *= 1000
        
        return data
    
    
    
    def met_data(self, data, level='sf'):
        level = str(level)
        if level == 'sf':
            result = self.surface[self.surface_dict[data], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            return Met.data_unit(result, data)
        
        elif level == 'all':
            result = self.upper[self.upper_dict[data],  :  ,self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1].copy()
            return Met.data_unit(result, data)
        
        else:
            pres   = pres_list.index(level)
            result = self.upper[self.upper_dict[data], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1].copy()
            return Met.data_unit(result, data)
        
    
    def wind_speed(self, level='sf'):
        level = str(level)
        if level == 'sf':
            u = self.surface[self.surface_dict['U10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.surface[self.surface_dict['V10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
        
        elif level == 'all':
            u = self.upper[self.upper_dict['u'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.upper[self.upper_dict['v'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
       
        else:
            pres = pres_list.index(level)
            u = self.upper[self.upper_dict['u'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.upper[self.upper_dict['v'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
        
        return np.sqrt(u**2 + v**2)
    
    # def vorticity(self, level='sf'):
    #     level = str(level)
        
    #     earth_radius = 6371e3  # in meters
    #     deg_to_rad = np.pi / 180

    #     # Pre-calculate deltas for longitude and latitude
    #     delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(self.lat_grid * deg_to_rad)
    #     delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(self.lat_grid)
        
    #     if level == 'all':
    #         u = self.upper[self.upper_dict['u'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #         v = self.upper[self.upper_dict['v'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #         vort = np.full_like(u, np.nan)
            
    #         for i in range(13):
    #             dv_dx = np.empty_like(v[i])
    #             dv_dx[:, 1:-1] = (v[i, :, 2:] - v[i, :, :-2]) / (2 * delta_lon[:, 1:-1])
    #             dv_dx[:, 0] = dv_dx[:, -1] = np.nan

    #             du_dy = np.empty_like(u[i])
    #             du_dy[1:-1, :] = (u[i, :-2, :] - u[i, 2:, :]) / (2 * delta_lat[1:-1, :])
    #             du_dy[0, :] = du_dy[-1, :] = np.nan

    #             # Calculate vorticity avoiding boundaries
    #             vort[i,1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]

    #         return vort
        
    #     else:
    #         if level == 'sf':
    #             u = self.surface[self.surface_dict['U10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #             v = self.surface[self.surface_dict['V10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            
    #         else:
    #             pres = pres_list.index(level)
    #             u = self.upper[self.upper_dict['u'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #             v = self.upper[self.upper_dict['v'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            
    #         vort = np.full_like(u, np.nan)
            
    #         dv_dx = np.empty_like(v)
    #         dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * delta_lon[:, 1:-1])
    #         dv_dx[:, 0] = dv_dx[:, -1] = np.nan

    #         du_dy = np.empty_like(u)
    #         du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * delta_lat[1:-1, :])
    #         du_dy[0, :] = du_dy[-1, :] = np.nan

    #         # Calculate vorticity avoiding boundaries
    #         vort[1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]
            
    #         return vort
        
    
    # def divergence(self, level='sf'):
    #     level = str(level)

    #     earth_radius = 6371e3  # meters
    #     deg_to_rad = np.pi / 180

    #     # Pre-calculate deltas for longitude and latitude
    #     delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(self.lat_grid * deg_to_rad)
    #     delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(self.lat_grid)

    #     if level == 'all':
    #         u = self.upper[self.upper_dict['u'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #         v = self.upper[self.upper_dict['v'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #         div = np.full_like(u, np.nan)

    #         for i in range(13):  # Assuming there are 13 vertical levels
    #             du_dx = np.empty_like(u[i])
    #             du_dx[:, 1:-1] = (u[i, :, 2:] - u[i, :, :-2]) / (2 * delta_lon[:, 1:-1])
    #             du_dx[:, 0] = du_dx[:, -1] = np.nan

    #             dv_dy = np.empty_like(v[i])
    #             dv_dy[1:-1, :] = (v[i, :-2, :] - v[i, 2:, :]) / (2 * delta_lat[1:-1, :])
    #             dv_dy[0, :] = dv_dy[-1, :] = np.nan

    #             # Calculate divergence avoiding boundaries
    #             div[i, 1:-1, 1:-1] = du_dx[1:-1, 1:-1] + dv_dy[1:-1, 1:-1]

    #         return div

    #     else:
    #         if level == 'sf':
    #             u = self.surface[self.surface_dict['U10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #             v = self.surface[self.surface_dict['V10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]

    #         else:
    #             pres = pres_list.index(level)  # assuming pres_list is defined with pressure levels
    #             u = self.upper[self.upper_dict['u'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
    #             v = self.upper[self.upper_dict['v'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]

    #         div = np.full_like(u, np.nan)

    #         du_dx = np.empty_like(u)
    #         du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * delta_lon[:, 1:-1])
    #         du_dx[:, 0] = du_dx[:, -1] = np.nan

    #         dv_dy = np.empty_like(v)
    #         dv_dy[1:-1, :] = (v[:-2, :] - v[2:, :]) / (2 * delta_lat[1:-1, :])
    #         dv_dy[0, :] = dv_dy[-1, :] = np.nan

    #         # Calculate divergence avoiding boundaries
    #         div[1:-1, 1:-1] = du_dx[1:-1, 1:-1] + dv_dy[1:-1, 1:-1]

    #         return div
        
    def calculate_derivative(self, data, axis, sigma=1.0):
        # Apply Gaussian filter for smoothing before derivative
        # smoothed = gaussian_filter1d(data, sigma=sigma, axis=axis, mode='nearest')
        smoothed = data
        derivative = np.gradient(smoothed, axis=axis)
        return derivative
    
    # def calculate_derivative(self, data, axis, sigma=1.0):
    #     # Apply Gaussian filter for smoothing before derivative
    #     smoothed = data

    #     # 4th order central difference implementation
    #     derivative = np.zeros_like(smoothed)
    #     if axis == 0:
    #         # Vertical (latitude)
    #         h = 1  # Assuming 1 as index spacing; modify as necessary for actual data spacing
    #         derivative[2:-2] = (-smoothed[0:-4] + 8 * smoothed[1:-3] - 8 * smoothed[3:-1] + smoothed[4:]) / (12 * h)
    #     else:
    #         # Horizontal (longitude)
    #         h = 1  # Assuming 1 as index spacing; modify as necessary for actual data spacing
    #         derivative[:, 2:-2] = (-smoothed[:, 0:-4] + 8 * smoothed[:, 1:-3] - 8 * smoothed[:, 3:-1] + smoothed[:, 4:]) / (12 * h)

        return derivative

    def vorticity(self, level = 'sf'):
        level = str(level)
        if level == 'sf':
            u = self.surface[self.surface_dict['U10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.surface[self.surface_dict['V10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]

        else:
            pres = pres_list.index(level)  # assuming pres_list is defined with pressure levels
            u = self.upper[self.upper_dict['u'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.upper[self.upper_dict['v'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
        
        earth_radius = 6371e3  # meters
        deg_to_rad = np.pi / 180

        # Pre-calculate deltas for longitude and latitude
        delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(self.lat_grid * deg_to_rad)
        delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(self.lat_grid)

        # Calculate derivatives
        dv_dx = self.calculate_derivative(v, 1) / delta_lon
        # Multiply by -1 to account for reverse latitude indexing
        du_dy = -self.calculate_derivative(u, 0) / delta_lat

        # Calculate vorticity
        vort = dv_dx - du_dy
        return vort

    def divergence(self, level = 'sf'):
        level = str(level)
        if level == 'sf':
            u = self.surface[self.surface_dict['U10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.surface[self.surface_dict['V10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]

        else:
            pres = pres_list.index(level)  # assuming pres_list is defined with pressure levels
            u = self.upper[self.upper_dict['u'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.upper[self.upper_dict['v'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
        
        earth_radius = 6371e3  # meters
        deg_to_rad = np.pi / 180

        # Pre-calculate deltas for longitude and latitude
        delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(self.lat_grid * deg_to_rad)
        delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(self.lat_grid)

        # Calculate derivatives
        du_dx = self.calculate_derivative(u, 1) / delta_lon
        # Multiply by -1 to account for reverse latitude indexing
        dv_dy = -self.calculate_derivative(v, 0) / delta_lat

        # Calculate divergence
        div = du_dx + dv_dy
        return div

@jit(nopython=True)
def calculate_bearing_position(lat, lon, bearing, distance):
    R = 6371.0  # Earth radius in kilometers
    
    # Convert latitude, longitude, and bearing to radians
    lat = radians(lat)
    lon = radians(lon)
    bearing = radians(bearing)
    
    # Calculate the new latitude
    new_lat = asin(sin(lat) * cos(distance / R) +
                   cos(lat) * sin(distance / R) * cos(bearing))
    
    # Calculate the new longitude
    new_lon = lon + atan2(sin(bearing) * sin(distance / R) * cos(lat),
                          cos(distance / R) - sin(lat) * sin(new_lat))
    
    # Convert the new latitude and longitude back to degrees
    new_lat = degrees(new_lat)
    new_lon = degrees(new_lon)
    
    return new_lat, new_lon     


#태풍 발생 위치 주변 & 10m/s 이상 지역 주변
def tc_finder(data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                   min_position, mask_size=2.5, init_size=2.5, local_min_size = 5, mslp_z_dis = 250 ,wind_thres=8, wind_field = 4 ,mslp_2hpa = 'n',back_prop = 'n'):
    
    tc_score = 0
    init_str = storm_time[0]
    mask_size = int(mask_size*4)
    local_min_size = int(local_min_size*4)+1
    
    if len(min_position)>0:
        mp_time = list(min_position.keys())
        mp_time.sort()
        
        if back_prop == 'n':
            last_key = mp_time[-1]

        else:
            last_key = mp_time[0]
            if pred_time >= last_key:
                return min_position
    

    # 해면기압의 로컬 최소값의 위치 찾기
    data_copy = np.copy(data)   #data_copy는 MSLP 정보
    filtered_data = minimum_filter(data_copy, size = local_min_size)
    local_minima = data_copy == filtered_data
    minima_labels, num_features = label(local_minima)
    minima_positions = np.array([np.mean(np.where(minima_labels == i), axis=1) for i in range(1, num_features+1)])

    #200-500hPa 층후값 로컬 최대값 위치 찾기
    z_minima = z_diff == maximum_filter(z_diff, size = local_min_size)
    z_labels, z_num_features = label(z_minima)
    z_positions = np.array([np.mean(np.where(z_labels == i), axis=1) for i in range(1, z_num_features+1)])
    
    
    
    #태풍 발생 시점 이전엔 찾기 X
    if back_prop == 'n':
        if pred_time < init_str:
            return min_position
    
    fm_positions = []

    # 각 z_position에 대해 모든 minima_positions까지의 거리를 계산 후 일정 거리 이내의 minima_position만 취득
    for z_pos in z_positions:
        for min_pos in minima_positions:
            z_pos = [int(z_pos[0]), int(z_pos[1])]
            min_pos = [int(min_pos[0]), int(min_pos[1])]
            # 유클리드 거리 계산
            distance = haversine_distance(lat_grid[z_pos[0], z_pos[1]], lon_grid[z_pos[0], z_pos[1]],
                                 lat_grid[min_pos[0], min_pos[1]],lon_grid[min_pos[0], min_pos[1]])

            if distance <= mslp_z_dis:
                fm_positions.append(min_pos)

    minima_positions = np.unique(fm_positions, axis=0) # 중복 위치 제거
    

    # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성 후 지우기
    # if back_prop == 'n':
    wind_mask = wind_speed >= wind_thres         
    expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((wind_field+1,wind_field+1)))  # wind_mask의 주변 2픽셀 확장
    data_copy[~expanded_wind_mask] = np.nan # 확장된 마스크를 사용하여 wind_speed > 10 조건과 그 주변 2픽셀 이외의 위치를 NaN으로 설정
        
    
    #처음엔 태풍 발생 위치 주변에서 찾기
    if len(min_position) < 1:
        if pred_time <= storm_time[0] + timedelta(days=2):
            data_copy[(lat_grid > (storm_lat[storm_time == pred_time]+init_size))|(lat_grid < (storm_lat[storm_time == pred_time]-init_size))] = np.nan   
            data_copy[(lon_grid > (storm_lon[storm_time == pred_time]+init_size))|(lon_grid < (storm_lon[storm_time == pred_time]-init_size))] = np.nan 
        else:
            if pred_time == storm_time[0] + timedelta(days=3):
                print(pred_time.strftime("%Y/%m/%d/%HUTC"), "태풍 발생 X.")
            return min_position
        
    # 태풍 발생 이후에는
    if min_position:
        last_min_idx = min_position[last_key]['idx']
        
        row_start = max(0, last_min_idx[0] - mask_size)
        row_end = min(data_copy.shape[0], last_min_idx[0] + mask_size + 1)  # +1은 Python의 슬라이싱이 상한을 포함하지 않기 때문
        col_start = max(0, last_min_idx[1] - mask_size)
        col_end = min(data_copy.shape[1], last_min_idx[1] + mask_size + 1)
        data_nan_filled = np.full(data_copy.shape, np.nan)
        data_nan_filled[row_start:row_end, col_start:col_end] = data_copy[row_start:row_end, col_start:col_end]
        data_copy = data_nan_filled
        # plt.imshow(data_copy)
        # plt.show()

    

    #data_copy의 모든 값들이 Nan이면 패스
    if np.isnan(data_copy).all():
        print(pred_time.strftime("%Y/%m/%d/%HUTC"), "모든 값이 NaN입니다. 유효한 최소값이 없습니다.")


    #data_copy에서 최소값 찾기
    else:
        #data_copy에서 nan이 아닌 부분에서만 minima_position 살리기
        filtered_positions = []
        for pos in minima_positions:
            lat, lon = int(pos[1]), int(pos[0])
            if not np.isnan(data_copy[lon, lat]):
                filtered_positions.append((int(lon), int(lat)))


        
        if mslp_2hpa == 'y':
            # if back_prop == 'n':
            for min_pos in filtered_positions:
                # print(lat_grid[min_pos[0], min_pos[1]], lon_grid[min_pos[0], min_pos[1]],data[min_pos[0], min_pos[1]])
                for bearing in np.arange(0,360,45):
                    
                    d=500
                    
                    try:
                        new_lat, new_lon = calculate_bearing_position(lat_grid[min_pos[0], min_pos[1]], lon_grid[min_pos[0], min_pos[1]], bearing, d)
                        new_lat = np.round(new_lat / 0.25) * 0.25
                        new_lon = np.round(new_lon / 0.25) * 0.25
                        # print((min_pos[0], min_pos[1]),(new_lat, new_lon))
                        # print(haversine((lat_indices[min_pos[0]], lon_indices[min_pos[1]]),(new_lat, new_lon),unit = 'km'))
                        mslp_diff = data[np.where(lat_indices == new_lat)[0], np.where(lon_indices == new_lon)[0]]-data[min_pos[0], min_pos[1]]
                        
                        # print(new_lat, new_lon, mslp_diff)
                        # ax.scatter(new_lon, new_lat, s=20, c='black')
                    #새 위치가 데이터 범위 벗어나는 것 무시
                    except IndexError:
                        continue
                    
                    if mslp_diff < 2:
                        filtered_positions.remove(min_pos)
                        break
        
                     
        minima_positions = filtered_positions  
        
        if (len(minima_positions) < 1) and (len(min_position)>0):  #태풍 소멸 이후
            if min_position[last_key]['type'] != 'ex':
                print(pred_time.strftime("%Y/%m/%d/%HUTC"), "태풍이 소멸하였습니다.")
                min_position[last_key]['type'] = 'ex'

                
            
        elif (len(minima_positions) < 1) and (len(min_position)<1): #태풍 발생을 못 찾음
            pass
        
        
        elif (len(minima_positions) > 0) and (len(min_position)>0):
            dis_pos_list=[]

            if min_position[last_key]['type'] != 'ex':
                for pos in minima_positions:

                    # print('pos', pos)
                    min_index = (pos[0], pos[1])
                    min_lat = lat_indices[lat_start + pos[0]]
                    min_lon = lon_indices[lon_start + pos[1]]
                    min_value = data_copy[pos[0],pos[1]]


                    # 여러 minima가 있는 경우 우열을 가림. 이전보다 더 먼 곳에 위치한 minima가 pop됨
                    dis = haversine((min_lat, min_lon), (min_position[last_key]['lat'], min_position[last_key]['lon']), unit = 'km')
                    dis_pos_list.append([min_lat, min_lon, min_index, min_value, dis])
                    if len(dis_pos_list)>1:
                        if dis_pos_list[-1][4] > dis_pos_list[-2][4]:
                            dis_pos_list.pop()
                
                min_lat, min_lon, min_index, min_value = dis_pos_list[0][0], dis_pos_list[0][1], dis_pos_list[0][2], dis_pos_list[0][3]
                min_position[pred_time] = {'lon': min_lon, 'lat': min_lat, 'idx': min_index, 
                                                                    'mslp':  min_value, 'type':'tc'}
                if back_prop == 'y':
                    min_position[pred_time]['type'] = 'td'
            else:
                pass
                
        elif (len(minima_positions) > 0) and (len(min_position)<1):
            print(pred_time.strftime("%Y/%m/%d/%HUTC"), "태풍 발생")
            min_index = minima_positions[0]
            min_value = data_copy[int(minima_positions[0][0]),int(minima_positions[0][1])]
            min_lat = lat_indices[lat_start + min_index[0]]
            min_lon = lon_indices[lon_start + min_index[1]]


            min_position[pred_time] = {'lon': min_lon, 'lat': min_lat, 'idx': min_index, 
                                                                    'mslp':  min_value, 'type':'tc'}

            if back_prop == 'y':
                min_position[pred_time]['type'] = 'td'
        
    return min_position



#수증기 색상 함수
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name, cmap(np.linspace(minval, maxval, n)))
    return new_cmap

truncated_BrBG = truncate_colormap("BrBG", minval=0.35, maxval=1.0) #수증기 colormap 지정

# 새 컬러맵 생성: 확률이 0인 곳은 투명, 그 이상은 불투명
jet = matplotlib.colormaps['jet']   
newcolors = jet(np.linspace(0.3, 1, 256))
newcolors[:2, -1] = 0  # 첫 번째 색상을 완전 투명하게 설정
jet0 = LinearSegmentedColormap.from_list('TransparentJet', newcolors)


#점이 아닌 선의 색상으로 강도를 나타내는 함수
def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1.0, zorder=5, label = None):
    # x, y는 선의 좌표, z는 색상에 사용될 값
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    
    # z 값을 정규화
    z = np.asarray(z)

    # 선분을 색상으로 구분하여 그리기
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, zorder=zorder, label = label)
    
    ax.add_collection(lc)
    
    return lc

#ax 배경 지정
def setup_map(ax, back_color = 'y'):
    
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax.coastlines()
    if back_color == 'y':
        ocean_color = mcolors.to_rgba((147/255, 206/255, 229/255))
        land_color = mcolors.to_rgba((191/255, 153/255, 107/255))
        ax.add_feature(cfeature.OCEAN, color=ocean_color)
        ax.add_feature(cfeature.LAND, color=land_color, edgecolor='none')

def set_map(row = 1, col = 1, size = (10,8) ,back_color = 'n', proj  = ccrs.PlateCarree(0), extent = None):
    fig, axs = plt.subplots(row, col, figsize = size, subplot_kw={'projection': proj})

    if row == 1 and col == 1:
        gl = axs.gridlines(crs=ccrs.PlateCarree(0), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10)) 
        
        if extent != None:
            axs.set_extent(extent)
            
        axs.coastlines()
        if back_color == 'y':
            ocean_color = mcolors.to_rgba((147/255, 206/255, 229/255))
            land_color = mcolors.to_rgba((191/255, 153/255, 107/255))
            axs.add_feature(cfeature.OCEAN, color=ocean_color)
            axs.add_feature(cfeature.LAND, color=land_color, edgecolor='none')
    
    else:
        for ax in axs:
            gl = ax.gridlines(crs=ccrs.PlateCarree(0), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12}
            gl.ylabel_style = {'size': 12}
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10)) 
            
            if extent != None:
                ax.set_extent(extent)
                
            ax.coastlines()
            if back_color == 'y':
                ocean_color = mcolors.to_rgba((147/255, 206/255, 229/255))
                land_color = mcolors.to_rgba((191/255, 153/255, 107/255))
                ax.add_feature(cfeature.OCEAN, color=ocean_color)
                ax.add_feature(cfeature.LAND, color=land_color, edgecolor='none')
    
    return fig, axs

#contour 함수
def weather_map_contour(ax, lon_grid, lat_grid, data, hpa = 1000):
    
    if hpa == 1000:
        levels = np.arange(920, 1040, 4)
        bold_levels = np.arange(904,1033,16)
        levels = levels[~np.isin(levels, bold_levels)]
        filtered_data = ndimage.gaussian_filter(data, sigma=3, order=0)
        cs = ax.contour(lon_grid, lat_grid, filtered_data / 100, levels=levels, colors='black', transform=proj)
        ax.clabel(cs, cs.levels,inline=True, fontsize=10)
        cs_bold = ax.contour(lon_grid, lat_grid, filtered_data / 100, levels=bold_levels, colors='black', transform=proj, linewidths=3)
        ax.clabel(cs_bold, cs_bold.levels, inline=True, fontsize=10)
    
    elif hpa == 500:
        levels = np.arange(5220,6001,60)
        cs = ax.contour(lon_grid, lat_grid, data, levels=levels, colors='black', transform=proj)
        ax.clabel(cs, cs.levels,inline=True, fontsize=10)        
            

def contourf_and_save(ax, fig, lon_grid, lat_grid, data, min_position, 
                      title='', label='', levels=None, cmap='jet', save_path=''):
    
    contourf = ax.contourf(lon_grid, lat_grid, data, cmap=cmap, levels=levels, extend='both')
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
    cbar.set_label(label, fontsize=16)
    ax.set_title(title, fontsize=20, loc = 'right')
    if min_position:
        ax.set_title(title+f' ({min_position[list(min_position.keys())[-1]]["mslp"]:.0f}hPa)', fontsize=20, loc = 'right')
    plt.savefig(save_path, bbox_inches='tight')
    cbar.remove()
    contourf.remove()  # 이 방법으로 contourf 객체를 제거

@jit(nopython=True)
def ep_t(T, P, r):
    P0 = 1000  # 기준 기압(hPa)
    Rd = 287  # 건조 공기의 비열비(J/kgK)
    cpd = 1004  # 건조 공기의 정압 비열(J/kgK)
    Lv = 2.5e6  # 물의 증발열(J/kg)
    # P를 hPa에서 Pa로 변환
    # P = P * 100  
    r = r/1000
    # theta_e = T * (P0 / (P / 100))**(Rd / cpd) * np.exp((Lv * r/1000) / (cpd * T))
    theta_e = (T+Lv/cpd*r)*(P0/P)**(Rd/cpd)
    return theta_e




@jit(nopython=True)
def concentric_circles(lat, lon, distances, bearings):
    lat_c = np.empty((len(distances), len(bearings)), dtype=np.float32)
    lon_c = np.empty((len(distances), len(bearings)), dtype=np.float32)
    # w_t = np.empty((13, len(distances), len(bearings)), dtype=np.float32)
    # w_r = np.empty((13, len(distances), len(bearings)), dtype=np.float32)
    for i, distance in enumerate(distances):
        for j, bearing in enumerate(bearings):
            lat2, lon2 = calculate_bearing_position(lat, lon, distance, bearing)
            lat_c[i, j] = lat2
            lon_c[i, j] = lon2
            # w_t[:, i, j] = -u * np.sin(bearing) + v * np.cos(bearing)
            # w_r[:, i, j] = u * np.cos(bearing) + v * np.sin(bearing)
    return lat_c, lon_c

@jit(nopython=True)
def interpolate_data(data, lat_indices, lon_indices, lat_c, lon_c):
    data_ip = np.empty((data.shape[0], data.shape[1], lat_c.shape[0], lat_c.shape[1]))
    
    for i in range(lat_c.shape[0]):
        for j in range(lat_c.shape[1]):
            lat_idx = np.argsort(np.abs(lat_indices-lat_c[i][j]))[:2]
            lon_idx = np.argsort(np.abs(lon_indices-lon_c[i][j]))[:2]
            
            #합칠 데이터
            sum_data = np.zeros((data.shape[0], data.shape[1]))
            sum_dis = 0
            
            #만약 거리가 0이면 지정할 데이터, sign이 y로 바뀌면 data_0으로 지정
            data_0 = np.zeros((data.shape[0], data.shape[1]))
            sign_0 = 'n'
            
            for m in range(2):
                for n in range(2):
                    
                    mini_data = data[:, :, lat_idx[m], lon_idx[n]]
                    mini_dis = haversine_distance(lat_indices[lat_idx[m]], lon_indices[lon_idx[n]], lat_c[i][j], lon_c[i][j])
                    
                    
                    if mini_dis != 0:
                        sum_data += mini_data/mini_dis
                        sum_dis  += 1/mini_dis

                    else:
                        data_0 = mini_data.astype(np.float64)
                        sign_0 = 'y'


                if sign_0 == 'n':
                    data_ip[:, :, i, j] = sum_data / sum_dis
                else:
                    data_ip[:, :, i, j] = data_0

    return data_ip


class WindFieldSolver:
    
    #######################################################################
    # Get TY wind field solving Poisson eq
    # Presume source term from average relative vort, divergence in TY area
    #######################################################################
    def __init__(self, lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850, radius=333, dx=111e3/4, dy=111e3/4, dis_cal_sign = 'y', final_mask = None):
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.vort = vort
        self.div = div
        self.vort_850 = vort_850
        self.dis_cal_sign = dis_cal_sign
        self.final_mask = final_mask
        self.R = 6371  # Earth's radius in kilometers
        
    
    @staticmethod
    @jit(nopython=True)
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1 = [lat1, lon1]/180*np.pi
        lat2, lon2 = [lat2, lon2]/180*np.pi
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = 6371 * c
        return distance
    
    
    def find_vort_center(self, local_min_size=10):
        vort_max = self.vort_850 == maximum_filter(self.vort_850, size = local_min_size)
        v_labels, v_num_features = label(vort_max)
        v_positions = np.array([np.mean(np.where(v_labels == i), axis=1) for i in range(1, v_num_features+1)]).astype(int)
        dis_dict = {}
        for i in range(len(v_positions)):
            v_pos = v_positions[i]
            dis_dict[i] = haversine_distance(self.center_lat, self.center_lon, self.lat_grid[v_pos[0], v_pos[1]], self.lon_grid[v_pos[0], v_pos[1]])
        # min_v_pos = min(dis_dict, key=dis_dict.get)
        min_v_pos = v_positions[min(dis_dict, key=dis_dict.get)]
        return self.lat_grid[min_v_pos[0], min_v_pos[1]], self.lon_grid[min_v_pos[0], min_v_pos[1]]
        

    # def create_source_term(self, data):
    #     # distances = np.ones_like(self.lat_grid)*20000
    #     center_lat_array = np.ones_like(self.lat_grid)*self.center_lat
    #     center_lon_array = np.ones_like(self.lon_grid)*self.center_lon
        
    #     # lat_min = self.center_lat - 5
    #     # lat_max = self.center_lat + 5
    #     # lon_min = self.center_lon - 5
    #     # lon_max = self.center_lon + 5

    #     # # Find indices where the latitude and longitude fall within the 5-degree bounding box
    #     # mask_idx = (self.lat_grid >= lat_min) & (self.lat_grid <= lat_max) & (self.lon_grid >= lon_min) & (self.lon_grid <= lon_max)
        
        
        
    #     # for i in range(self.lat_grid.shape[0]):
    #     #     for j in range(self.lat_grid.shape[1]):
    #     # distances_part = haversine_distance(center_lat_array[mask_idx], center_lon_array[mask_idx], self.lat_grid[mask_idx], self.lon_grid[mask_idx])
    #     distances = haversine_distance(center_lat_array, center_lon_array, self.lat_grid, self.lon_grid)
    #     # print(distances_part)
    #     # distances[mask_idx] = distances_part
        
    #     mask = distances <= self.radius
    #     # print(distances[mask])
    #     mean_value = np.mean(data[mask]) if np.any(mask) else 0

    #     b = np.zeros_like(data)
    #     b[mask] = mean_value
    #     return b
    
    def create_source_term(self):
        if self.dis_cal_sign == 'y':
            # Create arrays for center latitude and longitude
            center_lat_array = np.ones_like(self.lat_grid) * self.center_lat
            center_lon_array = np.ones_like(self.lon_grid) * self.center_lon

            # Calculate absolute differences in degrees
            lat_diff = np.abs(self.lat_grid - self.center_lat)
            lon_diff = np.abs(self.lon_grid - self.center_lon)
            
            # Create a mask for points within 5 degrees
            within_5_degrees_mask = (lat_diff <= 5) & (lon_diff <= 5)
            
            # Calculate distances using the Haversine formula
            # distances = haversine_distance(center_lat_array, center_lon_array, self.lat_grid, self.lon_grid)
            distances = np.full_like(self.lat_grid, fill_value=np.nan, dtype=np.float64)
            if np.any(within_5_degrees_mask):
                distances[within_5_degrees_mask] = haversine_distance(
                    center_lat_array[within_5_degrees_mask], 
                    center_lon_array[within_5_degrees_mask], 
                    self.lat_grid[within_5_degrees_mask], 
                    self.lon_grid[within_5_degrees_mask]
                )
            # Define the maximum distance to calculate intervals dynamically
            # max_distance = 800  # Adjust this value as needed
            # interval_size = 50
            # intervals = [(i, i + interval_size) for i in range(0, max_distance, interval_size)]
            
            # previous_average = None
            # optimal_radius = None
            vort_list = []
            # interval_sign = 0
            # # Calculate average values for each interval and check for increasing pattern
            # for i, (low, high) in enumerate(intervals):
            #     mask = (distances > low) & (distances <= high)
            #     if np.any(mask):
            #         masked_sf = self.vort_850[mask]
                    
            #         current_average = np.mean(masked_sf)
            #         vort_list.append([high, np.mean(self.vort[mask]), np.std(self.vort[mask])])
                    
            #         # Check if the average is increasing
            #         if previous_average is not None and current_average > previous_average and interval_sign == 0:
            #             optimal_radius = intervals[i-1][1]  # The upper bound of the previous interval
            #             interval_sign +=1
            #             # break
            #         previous_average = current_average

            # # If no increase is found, consider the last interval's upper bound as optimal radius
            # if optimal_radius is None:
            #     optimal_radius = intervals[-1][1]

            # Create final mask based on optimal radius and apply to data
            optimal_radius = 333
            final_mask = distances <= optimal_radius
            
        else:
            final_mask = self.final_mask
            # vort_list = []
            
        sf_data = np.zeros_like(self.vort)
        vp_data = np.zeros_like(self.div)
        sf_data[final_mask] = self.vort[final_mask]
        vp_data[final_mask] = self.div[final_mask]
        # print(optimal_radius)
        # vort_list = np.array(vort_list)
        return sf_data, vp_data, final_mask
        
    @staticmethod
    @jit(nopython=True)
    def solve_poisson_with_scaling(lat_grid, b, dx, dy, scaling_factors):
        p = np.zeros_like(lat_grid)
        for _ in range(1000):  # Number of iterations for convergence
            p_old = p.copy()
            for i in range(1, lat_grid.shape[0]-1):
                for j in range(1, lat_grid.shape[1]-1):
                    p[i, j] = ((p_old[i+1, j] + p_old[i-1, j]) * dy**2 +
                               (p_old[i, j+1] + p_old[i, j-1]) * dx**2 * scaling_factors[i, j]**2 -
                               b[i, j] * dy**2 * dx**2 * scaling_factors[i, j]**2) / (2 * (dy**2 + dx**2 *scaling_factors[i, j]**2))
        return p
    
    def solve_poisson_4th_order(lat_grid, b, dx, dy, scaling_factors):
        p = np.zeros_like(lat_grid)
        dx *= scaling_factors
        for _ in range(1000):  # 수렴을 위한 반복 횟수
            p_old = p.copy()
            for i in range(2, lat_grid.shape[0]-2):
                for j in range(2, lat_grid.shape[1]-2):
                    # pxx = (-p_old[i+2, j] + 16*p_old[i+1, j] - 30*p_old[i, j] + 16*p_old[i-1, j] - p_old[i-2, j]) / (12 * dx**2 * scaling_factors[i, j]**2)
                    # pyy = (-p_old[i, j+2] + 16*p_old[i, j+1] - 30*p_old[i, j] + 16*p_old[i, j-1] - p_old[i, j-2]) / (12 * dy**2)
                    p[i, j] = (12 * b[i,j] * dx[i,j]**2 * dy[i,j]**2 - 
                               (dx[i,j]**2+dy[i,j]**2)*(-p_old[i+2, j] + 16*p_old[i+1, j]+ 16*p_old[i-1, j] - p_old[i-2, j]
                                -p_old[i, j+2] + 16*p_old[i, j+1] + 16*p_old[i, j-1] - p_old[i, j-2]))/30*(dx[i,j]**2+dy[i,j]**2)
        return p

    def compute_wind_field(self, psi, type='sf'):
        if type == 'vp':
            u = np.gradient(psi, axis=1) / (np.cos(np.radians(self.lat_grid)) * self.dx)
            v = -np.gradient(psi, axis=0) / self.dy
        else:
            u = np.gradient(psi, axis=0) / self.dy
            v = np.gradient(psi, axis=1) / (np.cos(np.radians(self.lat_grid)) * self.dx)
        return u, v

    def solve(self):
        scaling_factors = np.cos(np.radians(self.lat_grid))
        self.center_lat, self.center_lon = self.find_vort_center()
        sf_source, vp_source, final_mask = self.create_source_term()
        sf = self.solve_poisson_with_scaling(self.lat_grid, sf_source, self.dx, self.dy, scaling_factors)
        vp = self.solve_poisson_with_scaling(self.lat_grid, vp_source, self.dx, self.dy, scaling_factors)
        u_sf, v_sf = self.compute_wind_field(sf, 'sf')
        u_vp, v_vp = self.compute_wind_field(vp, 'vp')
        u = u_sf + u_vp
        v = v_sf + v_vp
        # return u, v, u_sf, v_sf, u_vp, v_vp
        # return u, v, sf_source, vort_list, optimal_radius, self.center_lat, self.center_lon
        return u, v, final_mask
    


from collections import defaultdict
from scipy.spatial.distance import cdist

def find_large_groups(mid_pos, ens_num_list, tar_pos, nearby_sign, distance_threshold, rank, min_size):
    distances = cdist(mid_pos, mid_pos)
    close_pairs = np.where(distances <= distance_threshold)

    groups = defaultdict(list)
    for i, j in zip(*close_pairs):
        # if i != j:
        groups[i].append(ens_num_list[j])
    # print(groups)
    
    
    value_to_keys = defaultdict(list)
    for key, value in groups.items():
        # 리스트를 튜플로 변환하여 가변 타입을 불변 타입으로 만들어 키로 사용
        value_to_keys[tuple(value)].append(key)

    # 중복된 value를 가진 key 중 하나만 남기고 나머지 제거
    for key_list in value_to_keys.values():
        if len(key_list) > 1:
            # 첫 번째 키를 제외하고 나머지 키들을 제거 대상 목록에 추가
            for key in key_list[1:]:  # 첫 번째 키는 남기고, 나머지 키들을 제거
                del groups[key]
    
    # 값의 길이에 따라 그룹 정렬 및 순위 지정
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    sorted_groups = [group for group in sorted_groups if len(group[1]) >= min_size]
    # 원하는 순위의 그룹 추출
    if len(sorted_groups) >= rank:
        target_group = sorted_groups[rank - 1][1]
        # print(target_group)
        group_indices = np.array([ens_num_list.index(member) for member in target_group if member in ens_num_list])
        # print(group_indices)
        group_tar_pos = tar_pos[group_indices]
        group_mid_pos = mid_pos[group_indices]
        
        return group_tar_pos, group_mid_pos, group_indices
        
    else:
        return None