#%%
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
import tcmarkers

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, minimum_filter, maximum_filter, label
from skimage.measure import regionprops

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



def storm_info(storm_name, storm_year, datetime_list=None, wind_thres=35):
    file_path = f'/Data/home/jjoo0727/Pangu-Weather/storm_info/{storm_year}_{storm_name}.csv'
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

@jit(nopython=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

#output 기상 정보 클래스
class Met:
    def __init__(self, output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid):
        self.surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        self.upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
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
    
    def vorticity(self, level='sf'):
        level = str(level)
        
        earth_radius = 6371e3  # in meters
        deg_to_rad = np.pi / 180

        # Pre-calculate deltas for longitude and latitude
        delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(self.lat_grid * deg_to_rad)
        delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(self.lat_grid)
        
        if level == 'all':
            u = self.upper[self.upper_dict['u'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            v = self.upper[self.upper_dict['v'], :, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            vort = np.full_like(u, np.nan)
            
            for i in range(13):
                dv_dx = np.empty_like(v[i])
                dv_dx[:, 1:-1] = (v[i, :, 2:] - v[i, :, :-2]) / (2 * delta_lon[:, 1:-1])
                dv_dx[:, 0] = dv_dx[:, -1] = np.nan

                du_dy = np.empty_like(u[i])
                du_dy[1:-1, :] = (u[i, :-2, :] - u[i, 2:, :]) / (2 * delta_lat[1:-1, :])
                du_dy[0, :] = du_dy[-1, :] = np.nan

                # Calculate vorticity avoiding boundaries
                vort[i,1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]

            return vort
        
        else:
            if level == 'sf':
                u = self.surface[self.surface_dict['U10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
                v = self.surface[self.surface_dict['V10'], self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            
            else:
                pres = pres_list.index(level)
                u = self.upper[self.upper_dict['u'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
                v = self.upper[self.upper_dict['v'], pres, self.lat_start:self.lat_end + 1, self.lon_start:self.lon_end + 1]
            
            vort = np.full_like(u, np.nan)
            
            dv_dx = np.empty_like(v)
            dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * delta_lon[:, 1:-1])
            dv_dx[:, 0] = dv_dx[:, -1] = np.nan

            du_dy = np.empty_like(u)
            du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * delta_lat[1:-1, :])
            du_dy[0, :] = du_dy[-1, :] = np.nan

            # Calculate vorticity avoiding boundaries
            vort[1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]
            
            return vort
        
    
    # def calculate_divergence(u, v, lat_grid, lon_grid):
    #     # 위도와 경도의 차이를 계산하기 위한 준비
    #     lat_diff = np.diff(lat_grid, axis=0)
    #     lon_diff = np.diff(lon_grid, axis=1)
        
    #     # 위도와 경도에 따른 실제 거리 계산
    #     lat_distances = haversine(lat_grid[:-1, :], lon_grid[:-1, :], lat_grid[1:, :], lon_grid[1:, :])
    #     lon_distances = haversine(lat_grid[:, :-1], lon_grid[:, :-1], lat_grid[:, 1:], lon_grid[:, 1:])
        
    #     # u와 v의 공간적 변화율 계산
    #     du_dx = np.diff(u, axis=1) / lon_distances
    #     dv_dy = np.diff(v, axis=0) / lat_distances
        
    #     # 경도 방향의 평균을 취하여 그리드 크기를 맞춤
    #     du_dx_avg = (du_dx[:, :-1] + du_dx[:, 1:]) / 2
    #     # 위도 방향의 평균을 취하여 그리드 크기를 맞춤
    #     dv_dy_avg = (dv_dy[:-1, :] + dv_dy[1:, :]) / 2
        
    #     # 수평 발산 계산
    #     divergence = du_dx_avg + dv_dy_avg
        
    #     return divergence

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
                   min_position, mask_size=2.5, init_size=2.5, local_min_size = 5, mslp_z_dis = 250 ,back_prop = 'n'):
    
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
                
    
        
    #pred_str 전의 시간의 태풍 정보만 불러옴
    storm_lon = storm_lon[storm_time<=pred_time]
    storm_lat = storm_lat[storm_time<=pred_time]
    storm_mslp = storm_mslp[storm_time<=pred_time]
    storm_time = storm_time[storm_time<=pred_time]
    

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
    
    '''mslp, z 그리기'''
    # if back_prop == 'y':
        # lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
        # lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
        
        # x_pos = np.array([int(i[0]) for i in minima_positions])
        # y_pos = np.array([int(i[1]) for i in minima_positions])
        # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        # setup_map(ax,back_color='n')
        # ax.set_extent(extent, crs=proj)
        # ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} MSLP', fontsize=20)
        # mslp_ax = ax.scatter(lon_grid, lat_grid, c=data)
        # plt.colorbar(mslp_ax, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
        # ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
        # plt.show()
        
        # x_pos = np.array([int(i[0]) for i in z_positions])
        # y_pos = np.array([int(i[1]) for i in z_positions])
        # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        # setup_map(ax,back_color='n')
        # ax.set_extent(extent, crs=proj)
        # ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} Z_diff', fontsize=20)
        # z_ax = ax.scatter(lon_grid, lat_grid, c=z_diff)
        # plt.colorbar(z_ax, ax=ax, orientation='vertical', label='Z_diff (m)', shrink=0.8)
        # ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
        # plt.show()
    
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

            # 거리가 225km 이하인 경우만 선택
            if distance <= mslp_z_dis:
                fm_positions.append(min_pos)

    minima_positions = np.unique(fm_positions, axis=0) # 중복 위치 제거
    

    # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성 후 지우기
    if back_prop == 'n':
        wind_mask = wind_speed >= 10         
        expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((9*2+1,9*2+1)))  # wind_mask의 주변 2픽셀 확장
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

    # if back_prop == 'y':
    #     lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
    #     lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
        
    #     x_pos = np.array([int(i[0]) for i in minima_positions])
    #     y_pos = np.array([int(i[1]) for i in minima_positions])
    #     fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    #     setup_map(ax,back_color='n')
    #     ax.set_extent(extent, crs=proj)
    #     ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} MSLP', fontsize=20)
    #     mslp_ax = ax.scatter(lon_grid, lat_grid, c=data_copy)
    #     plt.colorbar(mslp_ax, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
    #     ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
    #     plt.show()

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


        #주변 기압 2hPa보다 낮은지 확인

        # lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
        # lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
        
        # x_pos = np.array([int(i[0]) for i in minima_positions])
        # y_pos = np.array([int(i[1]) for i in minima_positions])
        # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        # setup_map(ax,back_color='n')
        # ax.set_extent(extent, crs=proj)
        # ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} MSLP', fontsize=20)
        # mslp_ax = ax.scatter(lon_grid, lat_grid, c=data, cmap = 'jet')
        # plt.colorbar(mslp_ax, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
        # ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
        # plt.show()
        
        if back_prop == 'n':
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
def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1.0, zorder=5):
    # x, y는 선의 좌표, z는 색상에 사용될 값
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    
    # z 값을 정규화
    z = np.asarray(z)

    # 선분을 색상으로 구분하여 그리기
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
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


# %%
#위경도 지정
lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

#태풍 지정
storm_name = 'hinnamnor'                                                                               
storm_name = storm_name.upper()
storm_year = 2022

#예측 시간 지정, 초기 시간 지정, 앙상블 수
predict_interval_list = np.arange(0,24*7+1,6)
first_str = '2022/08/27/00UTC'   
ens_num = 100

surface_factors = []  # 예시: 지표면에서는 'MSLP'만 선택
upper_factors = ['z'] 
perturation_scale = 0.05


surface_factors.sort()
upper_factors.sort()
surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가


first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
datetime_list = np.array([first_time + timedelta(hours=int(hours)) for hours in predict_interval_list])
# datetime_array = np.array([(first_time + timedelta(hours=int(hours))) for hours in predict_interval_list])

storm_lon, storm_lat, storm_mslp, storm_time = storm_info(storm_name, storm_year, datetime_list = datetime_list, wind_thres=0)   #태풍 영문명, 년도 입력

min_position = {}  # 태풍 중심 정보 dictionary



# for ens in range(ens_num):
for ens in range(ens_num):
    print(f'{ens}번째 앙상블 예측')
    min_position[ens] = {}
    output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
    
    
    for predict_interval in predict_interval_list:
        predict_time = first_time + timedelta(hours=int(predict_interval))
        predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        mslp = met.met_data('MSLP')
        wind_speed = met.wind_speed()
        z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
        

        min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                min_position[ens], mask_size = 2.5, init_size=5, local_min_size = 5, mslp_z_dis = 250)


    for predict_interval in predict_interval_list[::-1]:
        predict_time = first_time + timedelta(hours=int(predict_interval))
        predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        mslp = met.met_data('MSLP')
        wind_speed = met.wind_speed()
        z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
        
        min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                    wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                    min_position[ens], mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 250)
        
        min_position[ens] = {k: min_position[ens][k] for k in sorted(min_position[ens])}
#%%
min_position[0]
#%%
mp_key = [k for k in min_position if min_position[k] != {}]
print(len(mp_key))

first_time_dict = {}
for key in mp_key:
    time_list = list(min_position[key].keys())
    if time_list[0] in first_time_dict:
        first_time_dict[time_list[0]] += 1
    else:
        first_time_dict[time_list[0]] = 1
        
first_time_dict_key = list(first_time_dict.keys())
first_time_dict_key.sort()
first_time_dict = {key: first_time_dict[key] for key in first_time_dict_key}
first_time_dict
#%%
# for ens in range(ens_num):
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.set_title(f'{first_str} (+{predict_interval_list[-1]}h)\n{predict_str}', fontsize=20, loc = 'left')
ax.set_title(f'ENS{surface_str}{upper_str}{perturation_scale} Track\n{storm_name}', fontsize=20, loc = 'right')
ax.set_extent(extent, crs=proj)
setup_map(ax)

ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=3)
model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=3)
plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)



for i in range(len(storm_time)):
    new_time = storm_time[i].strftime("%Y/%m/%d/%HUTC")
    if new_time.endswith('00UTC'):
        dx, dy = 3, -3  # 시간 나타낼 위치 조정
        new_lon, new_lat = storm_lon[i] + dx, storm_lat[i] + dy
        
        # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
        ax.text(storm_lon[i], storm_lat[i], new_time[8:-6]
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)


# for ens in [77]:
for ens in range(ens_num):
# for ens in group2:

    lons = [pos['lon'] for _,pos in min_position[ens].items()]
    lats = [pos['lat'] for _,pos in min_position[ens].items()]
    min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
    pred_times = [pos for pos,_ in min_position[ens].items()]
    # print(ens)
    lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)

    for i in range(len(pred_times)):
        if pred_times[i].hour == 0:
            ax.text(lons[i],lats[i], str(pred_times[i].day)
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder = 6)

    if ens == 0:
        ax.text(lons[-1],lats[-1], '0 ENS'
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)

    
ax.legend(loc='upper right')


lons_all = np.concatenate([np.array([pos['lon'] for _, pos in min_position[ens].items()]) for ens in range(ens_num)])
lats_all = np.concatenate([np.array([pos['lat'] for _, pos in min_position[ens].items()]) for ens in range(ens_num)])


xy = np.vstack([lons_all, lats_all])
kde = gaussian_kde(xy)
positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
f = np.reshape(kde(positions).T, lon_grid.shape)


levels = np.linspace(0, 0.015, 100)
cf = ax.contourf(lon_grid, lat_grid, f, levels=levels, transform=proj, cmap=jet0, extend = 'both')
plt.show()


#%%
#그룹 찾기, 그룹 별 세력 표시
lat_values = []

for ens in range(100):
    for key, pos in min_position[ens].items():
        if key.strftime("%Y/%m/%d/%HUTC") == '2022/09/01/00UTC':
            lat_values.append([ens, pos['lat']])  # (ens index, lat)



# 위도(lat) 값이 가장 높은 ens와 가장 낮은 ens를 찾음
lat_values.sort(key=lambda x: x[1])
# lat_values2.sort(key=lambda x: x[1])
group1 = [item[0] for item in lat_values[:10]]
group2 = [item[0] for item in lat_values[-10:]]



for ens in range(ens_num):
    min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
    x = list(min_position[ens].keys())
    plt.plot(x, min_values, color = 'black')
    
# group1의 첫 번째 원소에만 레이블을 추가합니다.
first = True
for ens in group1:
    min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
    x = list(min_position[ens].keys())
    if first:
        plt.plot(x, min_values, color='red', label='group1')
        first = False
    else:
        plt.plot(x, min_values, color='red')

# group2의 첫 번째 원소에만 레이블을 추가합니다.
first = True
for ens in group2:
    min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
    x = list(min_position[ens].keys())
    
    if first:
        plt.plot(x, min_values, color='blue', label='group2')

        first = False
    else:
        plt.plot(x, min_values, color='blue')

plt.plot(storm_time, storm_mslp, color = 'green', label = 'Best track')
# Set the locator for the x-axis to locate days
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# Set the formatter for the x-axis to display dates in 'mm.dd' format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d'))

# Optionally, rotate the date labels for better readability
# plt.gcf().autofmt_xdate()
plt.ylabel('pressure(hPa)')
plt.xlabel('Time')
plt.legend(loc='lower left')

#%%
#group별 고도 별 평균 그리기

levels_dict = {
            't':{
                200: np.arange(220,231,1),
                500: np.arange(250,281,1),
                850: np.arange(280,301,1),
                925: np.arange(280,301,1)
            }

            ,
               
            'z':{200: np.arange(11760,12720,120),
               500: np.arange(5580,6000,60),
               850: np.arange(1380,1680,30),
               925: np.arange(600,840,30)    
               }}

levels_diff = {'t': np.arange(-2,2.1,0.1),
                'z': np.arange(-100,101,10)}

data_type_list = ['t','z']
wind_scale = 1000

path = f'/Data/home/jjoo0727/Pangu-Weather/plot/ENS/group_level_diff/{perturation_scale}ENS{surface_str}{upper_str}'
for data_type in levels_dict:
    for alt in levels_dict[data_type]:
        for predict_interval in np.arange(0,169,6):
            # print(data_t)
            levels = levels_dict[data_type][alt]
            group1_values = []
            group2_values = []
            predict_time = first_time + timedelta(hours=int(predict_interval))
            predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
            
            fig, ax = plt.subplots(1, 3, figsize=(10*latlon_ratio*3, 10), subplot_kw={'projection': proj})
            fig.suptitle(f'{predict_str} {alt}hPa height(m) & wind', fontsize=45)
        
            # Group 1
            for ens in group1:
                output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                z = met.met_data(data_type, level = alt)
                u = met.met_data('u', level = alt)
                v = met.met_data('v', level = alt)
                # ax.contour(lon_grid, lat_grid, z, colors='red', levels = levels, alpha=0.1)
                group1_values.append(np.array([z,u,v]))
                # print(ens)

            # Group 2
            for ens in group2:
                output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                z = met.met_data(data_type, level = alt)
                u = met.met_data('u', level = alt)
                v = met.met_data('v', level = alt)
                # ax.contour(lon_grid, lat_grid, z, colors='blue', levels = levels, alpha=0.1)
                group2_values.append(np.array([z,u,v]))
            
            group1_mean = np.mean(np.stack(group1_values), axis=0)
            group2_mean = np.mean(np.stack(group2_values), axis=0)


            for idx, axi in enumerate(ax.ravel()):
                axi.set_extent(extent, crs=proj)
                setup_map(axi, back_color='n')
                
                # 첫 번째 축에는 group1의 데이터를 그림
                if idx == 0:
                    axi.set_title('group1', fontsize=30)
                    g = axi.contourf(lon_grid, lat_grid, group1_mean[0], levels=levels, cmap = 'gist_ncar', extend = 'both')
                    Q = axi.quiver(lon_grid[::4,::4], lat_grid[::4,::4], group1_mean[1][::4,::4], group1_mean[2][::4,::4], scale=wind_scale)
                    axi.quiverkey(Q, X=0.93, Y=1.05, U=10, label='10 m/s', labelpos='E')
                    fig.colorbar(g, ax = axi, shrink=0.8)
                    
                # # 두 번째 축에는 group2의 데이터를 그림
                elif idx == 1:
                    axi.set_title('group2', fontsize=30)
                    g = axi.contourf(lon_grid, lat_grid, group2_mean[0], levels=levels, cmap = 'gist_ncar', extend = 'both')
                    Q = axi.quiver(lon_grid[::4,::4], lat_grid[::4,::4], group2_mean[1][::4,::4], group2_mean[2][::4,::4], scale=wind_scale)
                    axi.quiverkey(Q, X=0.93, Y=1.05, U=10, label='10 m/s', labelpos='E')
                    fig.colorbar(g, ax = axi, shrink=0.8)
                    
                elif idx == 2:
                    axi.set_title('group1-group2', fontsize=30)
                    g = axi.contourf(lon_grid, lat_grid, group1_mean[0]-group2_mean[0], levels=levels_diff[data_type], cmap = 'seismic', extend = 'both')
                    Q = axi.quiver(lon_grid[::4,::4], lat_grid[::4,::4], group1_mean[1][::4,::4] - group2_mean[1][::4,::4], group1_mean[2][::4,::4]-group2_mean[2][::4,::4], scale=wind_scale/5)
                    axi.quiverkey(Q, X=0.93, Y=1.05, U=10, label='10 m/s', labelpos='E')
                    fig.colorbar(g, ax = axi, shrink=0.8)
                    

            plt.subplots_adjust(wspace=0.05)        
            # 폴더 경로 생성
            folder_path = path+f'/{data_type}_{alt}hPa'

            # 해당 경로에 폴더가 없으면 생성
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            else:
                pass
            
            plt.savefig(f'{folder_path}/{predict_interval}')
            # plt.show()
            plt.close()




#%%
#group별 위도, 경도 평균 그리기
zonal_data = {}
merid_data = {}

lon_space = 7.5
lat_space = 2
lon_space, lat_space = int(lon_space*4), int(lat_space*4)

variable_keys = list(upper_dict.keys())
variable_keys.append('ept')

for pred_time in datetime_list[6:21]:
    print(pred_time)
    zonal_data[pred_time]={}
    merid_data[pred_time]={}
    
    for g_num, group in enumerate([group1, group2]):
        zonal_data[pred_time][g_num]={}
        merid_data[pred_time][g_num]={}
        
        for key in variable_keys:
            zonal_data[pred_time][g_num][key]=[]
            merid_data[pred_time][g_num][key]=[]
            
            for ens_num, ens in enumerate(group):
                # print(ens)
                c_lat = min_position[ens][pred_time]['idx'][0]
                c_lon = min_position[ens][pred_time]['idx'][1]
                
                predict_interval = pred_time - first_time   
                predict_interval = int(predict_interval.total_seconds() / 3600)
                
                output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                
                if key != 'ept':
                    variable = met.met_data(key, level = 'all')
                
                elif key == 'ept':
                    t = met.met_data('t', level = 'all')
                    q = met.met_data('q', level = 'all')
                    variable = ep_t(t, pres_array[:,np.newaxis,np.newaxis], q)
            
                zonal_data[pred_time][g_num][key].append(variable[:, c_lat - lat_space: c_lat + lat_space+1, c_lon - lon_space: c_lon + lon_space+1])
                merid_data[pred_time][g_num][key].append(variable[:, c_lat - lat_space: c_lat + lat_space+1, c_lon - lon_space: c_lon + lon_space+1])
                
            zonal_data[pred_time][g_num][key] = np.mean(np.array(zonal_data[pred_time][g_num][key]), axis=(0,2))
            merid_data[pred_time][g_num][key] = np.mean(np.array(merid_data[pred_time][g_num][key]), axis=(0,3))[:,::-1]



#%%
variable_dict = {
    't'  : {'title': 'Temperature anomaly', 'level': np.arange(-10,10,1), 'diff_level': np.arange(-10,10.5,1), 
            'cmap': 'seismic', 'bar_label': 'K'}, 
    'q'  : {'title': 'Specific Humidity', 'level': np.arange(0,20,0.5), 'diff_level': np.arange(-10,10.5,1), 
            'cmap': 'gist_ncar', 'bar_label': 'g/kg'}, 
    'ept': {'title': 'Equivalent Potential Temperature', 'level': np.arange(335,366,1), 'diff_level': np.arange(-20,20.5,1), 
            'cmap': 'gist_ncar', 'bar_label': 'K'},
    'z':   {'title': 'Geopotential Height anomaly', 'level': np.arange(-100,101,5), 'diff_level': np.arange(-200,200.1,10), 
            'cmap': 'seismic', 'bar_label': 'm'},
    'u'  : {'title': 'Zonal Wind(U)', 'level': np.arange(-50,51,1), 'diff_level': np.arange(-50,51,1),
            'cmap': 'seismic', 'bar_label': 'm/s'}, 
    'v'  : {'title': 'Meriodonal Wind(V)', 'level': np.arange(-50,51,1), 'diff_level': np.arange(-50,50.1,1),
            'cmap': 'seismic', 'bar_label': 'm/s'} 
    }

x = np.arange(-lon_space/4,lon_space/4+0.1, 0.25)
y = pres_array

xx, yy = np.meshgrid(x, y)
path = f'/Data/home/jjoo0727/Pangu-Weather/plot/ENS/group_zonal/{perturation_scale}ENS{surface_str}{upper_str}'
for pred_time in datetime_list[6:21]:
    pred_str = pred_time.strftime("%Y.%m.%d.%HUTC")

    for plot_num in range(2):    
        fig, axs = plt.subplots(3,3,figsize = (15,12))
        fig.suptitle(pred_str, y=0.95, fontsize = 25)
        
        key_list = list(variable_dict.keys())[plot_num*3:plot_num*3+3]
        # print(key_list)
        for i, g_num in enumerate(zonal_data[pred_time]):
            for j, key in enumerate(key_list):
                ax = axs[j,i]
                if key == 't' or key == 'z':
                    cax = ax.contourf(xx, yy, zonal_data[pred_time][g_num][key]-np.mean(zonal_data[pred_time][g_num][key], axis = 1)[:,np.newaxis], cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
                else:
                    cax = ax.contourf(xx, yy, zonal_data[pred_time][g_num][key], cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
                ax.set_xlabel('Longitude(°)')
                ax.set_ylabel('Pressure(hPa)')
                ax.invert_yaxis()
                cbar= plt.colorbar(cax, ax = ax, shrink = 1)
                cbar.set_label(variable_dict[key]['bar_label'], fontsize=15)
                ax.set_title(f'group {g_num+1}', fontsize = 14)
                ax.axhline(200, color='black', linestyle='--')
                
                if i==1:
                    ax.set_title(f'{variable_dict[key]["title"]}\ngroup {g_num+1}', fontsize = 14)
                
        for j, key in enumerate(key_list):
            ax = axs[j,2]
            cax = ax.contourf(xx, yy, zonal_data[pred_time][0][key] - zonal_data[pred_time][1][key], cmap = 'seismic', levels = variable_dict[key]['diff_level'], extend = 'both')
            ax.set_xlabel('Longitude(°)')
            ax.set_ylabel('Pressure(hPa)')
            ax.set_title(f'group1 - group2', fontsize = 14)
            ax.invert_yaxis()
            ax.axhline(200, color='black', linestyle='--')
            
            cbar= plt.colorbar(cax, ax = ax, shrink = 1)
            cbar.set_label(variable_dict[key]['bar_label'], fontsize=15)
            
        plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
        
        if not os.path.exists(f'{path}/{key_list}'):
            os.makedirs(f'{path}/{key_list}')
        else:
            pass  
        plt.savefig(f'{path}/{key_list}/{pred_str}.png')
        plt.close()


#%%
#Radial interpolation 진행
#변수별로 연산진행하는 것보다 하나의 배열로 합쳐서 하는 것이 3-4배 정도 빠름(numba 특성)
#for문 없앨때마다 연산이 기하급수적으로 빨라짐

pres_array = np.array(pres_list, dtype=np.float32)
rad_data = {}
distances = np.arange(0, 501, 5)
bearing = np.linspace(0, 358, 180)
rad_factor = upper_factor+['ept']


for g_num, group in enumerate([group1, group2]):
    rad_data[g_num]={}
    
    for ens_num, ens in enumerate(group):
        rad_data[g_num][ens_num]={}
        
        
        start = time.time()
        for mp_time, mp in min_position[ens].items():
            
            predict_interval = mp_time - first_time   
            predict_interval = int(predict_interval.total_seconds() / 3600)
            output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
            met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
            rad_data[g_num][ens_num][predict_interval]={}
            
            #일정 거리, 360도 각 방위에 떨어진 지점의 lat, lon을 lat_c, lon_c에 지정
            lat_c, lon_c = concentric_circles(mp['lat'], mp['lon'], distances, bearing)

            data = []
            
            #rad_factor에 지정한 변수 데이터를 하나의 data 배열에 저장
            for key in rad_factor:
                    
                if key in upper_factor:
                    data.append(met.met_data(key, level = 'all'))
                    
                elif key == 'ept':
                    t = met.met_data('t', level = 'all')
                    q = met.met_data('q', level = 'all')
                    data.append(ep_t(t, pres_array[:,np.newaxis,np.newaxis], q))
                    
                
            data = np.array(data)
            data_ip = interpolate_data(data, lat_indices, lon_indices, lat_c, lon_c)
            for i, key in enumerate(rad_factor): 
                rad_data[g_num][ens_num][predict_interval][key] = data_ip[i]

            
        end = time.time()
        print(f"group{g_num+1}, ens: {ens_num}, size: {len(min_position[ens])}, {end-start:.2f}s")


rad_mean = {}

for key in rad_factor:
    rad_mean[key] = {}

    for gp in range(2):
        rad_mean[key][gp] = {}

        for pi in np.arange(30,24*7+1,6):
            rad_mean[key][gp][pi] = []

            for ens in range(10):
                if pi in rad_data[gp][ens]:  # 키가 rad_data[0]에 존재하는지 확인
                    rad_mean[key][gp][pi].append(rad_data[gp][ens][pi][key])

            if rad_mean[key][gp][pi]==[]:
                del rad_mean[key][gp][pi]  # 비어있다면 해당 키 삭제
            
            else:
                rad_mean[key][gp][pi] = np.mean(rad_mean[key][gp][pi], axis=0)  


#%%

xx, yy = np.meshgrid(distances, pres_array)
path = f'/Data/home/jjoo0727/Pangu-Weather/plot/ENS/group_all_level_radial/{perturation_scale}ENS{surface_str}{upper_str}'
variable_dict = {
    'V_t': {'title': 'Tangential Wind', 'level': np.arange(-50,51,1), 'diff_level': np.arange(-30,31,1), 
            'cmap': 'seismic', 'bar_label': 'm/s'},  
    'V_r': {'title': 'Radial Wind', 'level': np.arange(-15,16,1), 'diff_level': np.arange(-5,6,1),
            'cmap': 'seismic', 'bar_label': 'm/s'}, 
    't'  : {'title': 'Temperature anomaly', 'level': np.arange(-5,5.5,0.5), 'diff_level': np.arange(-10,10.5,0.5), 
            'cmap': 'seismic', 'bar_label': 'K'}, 
    'ept': {'title': 'Equivalent Potential Temperature', 'level': np.arange(335,361,1), 'diff_level': np.arange(-10,10.5,0.5), 
            'cmap': 'gist_ncar', 'bar_label': 'K'},
    'u'  : {'title': 'Zonal Wind(U)', 'level': np.arange(-20,21,1), 'diff_level': np.arange(-10,11,0.5),
            'cmap': 'seismic', 'bar_label': 'm/s'}, 
    'v'  : {'title': 'Meriodonal Wind(V)', 'level': np.arange(-20,21,1), 'diff_level': np.arange(-10,10.1,0.5),
            'cmap': 'seismic', 'bar_label': 'm/s'}, 
    'z':   {'title': 'Geopotential Height anomaly', 'level': np.arange(-50,51,5), 'diff_level': np.arange(-50,50.1,5), 
            'cmap': 'seismic', 'bar_label': 'm'}
    }


for pi in rad_mean['t'][1]:
    
    if pi <= 120:
        group_dict = {}
        
        for gp in range(2):
            u = rad_mean['u'][gp][pi]
            v = rad_mean['v'][gp][pi]
            t = rad_mean['t'][gp][pi]
            q = rad_mean['q'][gp][pi]
            z = rad_mean['z'][gp][pi]
            ept = rad_mean['ept'][gp][pi]
            
            V_t = np.empty_like(u) 
            V_r = np.empty_like(v) 

            bearing_rad = np.radians(bearing)
            
            # 각 위치에 대한 접선 및 방사 성분 계산
            for i in range(len(bearing_rad)):
                V_r[:, :, i] =  u[:, :, i] * np.sin(bearing_rad[i]) + v[:, :, i] * np.cos(bearing_rad[i])
                V_t[:, :, i] = -u[:, :, i] * np.cos(bearing_rad[i]) + v[:, :, i] * np.sin(bearing_rad[i])
            
            group_dict[gp] = {'V_t': V_t, 'V_r': V_r, 't'  : t, 'ept': ept, 'u':u ,'v': v, 'z': z}

        for plot_num in range(3):
            fig, axs = plt.subplots(2,3,figsize=(25,15))
            predict_str = (first_time + timedelta(hours=int(pi))).strftime("%Y.%m.%d.%HUTC")
            fig.suptitle(f'{predict_str}(+{pi}h)', fontsize=35, y=0.97)

            key_list = list(variable_dict.keys())[plot_num*2:plot_num*2+2]
            
            for i, key in enumerate(key_list):   
                
                ax = axs[i,0]
                ax.set_title('group1', fontsize=25)
                if key == 't' or key == 'z':
                    cax =ax.contourf(xx, yy, np.mean(group_dict[0][key], axis=2) - np.mean(group_dict[0][key], axis=(1,2))[:,np.newaxis], cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
                else:
                    cax =ax.contourf(xx, yy, np.mean(group_dict[0][key], axis=2), cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
                    
                cbar = plt.colorbar(cax, ax = ax, orientation='vertical', ticks = variable_dict[key]['level'][::10])
                cbar.set_label(variable_dict[key]['bar_label'], fontsize=15)


                ax = axs[i,1]
                ax.set_title(f"{variable_dict[key]['title']}\n group2", fontsize=25)
                if key == 't' or key == 'z':
                    cax =ax.contourf(xx, yy, np.mean(group_dict[1][key], axis=2) - np.mean(group_dict[1][key], axis=(1,2))[:,np.newaxis], cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
                else:
                    cax =ax.contourf(xx, yy, np.mean(group_dict[1][key], axis=2), cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
                cbar = plt.colorbar(cax, ax = ax, orientation='vertical', ticks = variable_dict[key]['level'][::10])
                cbar.set_label(variable_dict[key]['bar_label'], fontsize=15)
                    
                    
                ax = axs[i,2]
                ax.set_title('group1 - group2', fontsize=25)
                cax =ax.contourf(xx, yy, np.mean(group_dict[0][key] - group_dict[1][key], axis=2), cmap = 'seismic', levels = variable_dict[key]['diff_level'], extend = 'both')
                cbar = plt.colorbar(cax, ax = ax, orientation='vertical', ticks = variable_dict[key]['diff_level'][::10])
                cbar.set_label(variable_dict[key]['bar_label'], fontsize=15)
                
                
                for j in range(3):
                    ax = axs[i,j]
                    ax.set_xlabel("distance(km)", fontsize = 18)
                    ax.set_ylabel('Pressure(hPa)', fontsize = 18)
                    ax.axhline(200, color='black', linestyle='--')
                    ax.invert_yaxis()
                    
                plt.subplots_adjust(hspace=0.4) 
            
            file_path = path+f'/{key_list}'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            else:
                pass  
             
            plt.savefig(f'{file_path}/{predict_str}.png')
            plt.close()
        
        
    
    

