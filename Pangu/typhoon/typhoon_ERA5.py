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

# lat_indices = np.linspace(90, -90, 721)
# lon_indices = np.linspace(-180, 180, 1441)[:-1]


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


#output 기상 정보 클래스
class Met:
    def __init__(self, output_data_dir, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid):
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
        ax.set_title(title+f' ({min_position[last_key]["mslp"]:.0f}hPa)', fontsize=20, loc = 'right')
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

def tc_finder(data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                   min_position, mask_size=2.5, init_size=2.5, local_min_size = 5, mslp_z_dis = 250 ,back_prop = 'n'):
    

    init_str = storm_time[0]
    mask_size = int(mask_size*4)
    local_min_size = int(local_min_size*4)
    
    if len(min_position)>0:
        mp_time = list(min_position.keys())
        mp_time.sort()
        
        if back_prop == 'n':
            last_key = mp_time[-1]
        else:
            last_key = mp_time[0]
    
    #pred_str 전의 시간의 태풍 정보만 불러옴
    storm_lon = storm_lon[storm_time<=pred_time]
    storm_lat = storm_lat[storm_time<=pred_time]
    storm_mslp = storm_mslp[storm_time<=pred_time]
    storm_time = storm_time[storm_time<=pred_time]
    print(np.min(data))
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
    #     lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
    #     lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
        
    #     x_pos = np.array([int(i[0]) for i in minima_positions])
    #     y_pos = np.array([int(i[1]) for i in minima_positions])
    #     fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    #     setup_map(ax,back_color='n')
    #     ax.set_extent(extent, crs=proj)
    #     ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} MSLP', fontsize=20)
    #     mslp_ax = ax.scatter(lon_grid, lat_grid, c=data)
    #     plt.colorbar(mslp_ax, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
    #     ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
    #     plt.show()
        
    #     x_pos = np.array([int(i[0]) for i in z_positions])
    #     y_pos = np.array([int(i[1]) for i in z_positions])
    #     fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    #     setup_map(ax,back_color='n')
    #     ax.set_extent(extent, crs=proj)
    #     ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} Z_diff', fontsize=20)
    #     z_ax = ax.scatter(lon_grid, lat_grid, c=z_diff)
    #     plt.colorbar(z_ax, ax=ax, orientation='vertical', label='Z_diff (m)', shrink=0.8)
    #     ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
    #     plt.show()
    
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
        data_copy[(lat_grid > (storm_lat[0]+init_size))|(lat_grid < (storm_lat[0]-init_size))] = np.nan   
        data_copy[(lon_grid > (storm_lon[0]+init_size))|(lon_grid < (storm_lon[0]-init_size))] = np.nan 
    
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

    if back_prop == 'y':
        lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
        lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
        
        x_pos = np.array([int(i[0]) for i in minima_positions])
        y_pos = np.array([int(i[1]) for i in minima_positions])
        fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        setup_map(ax,back_color='n')
        ax.set_extent(extent, crs=proj)
        ax.set_title(f'{pred_time.strftime("%Y/%m/%d/%HUTC")} MSLP', fontsize=20)
        mslp_ax = ax.scatter(lon_grid, lat_grid, c=data_copy)
        plt.colorbar(mslp_ax, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
        ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
        plt.show()

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

        for min_pos in filtered_positions:
            for bearing in np.arange(0,360,45):
                d=500
                
                try:
                    new_lat, new_lon = calculate_bearing_position(lat_grid[min_pos[0], min_pos[1]], lon_grid[min_pos[0], min_pos[1]], bearing, d)
                    new_lat = np.round(new_lat / 0.25) * 0.25
                    new_lon = np.round(new_lon / 0.25) * 0.25
                    # print((min_pos[0], min_pos[1]),(new_lat, new_lon))
                    # print(haversine((lat_indices[min_pos[0]], lon_indices[min_pos[1]]),(new_lat, new_lon),unit = 'km'))
                    mslp_diff = data[np.where(lat_indices == new_lat)[0], np.where(lon_indices == new_lon)[0]]-data[min_pos[0], min_pos[1]]
                

                #새 위치가 데이터 범위 벗어나는 것 무시
                except IndexError:
                    continue
                
                if mslp_diff < 2:
                    filtered_positions.remove(min_pos)
                    break
                        
        minima_positions = filtered_positions  
        print(minima_positions)    
        
        if (len(minima_positions) < 1) and (len(min_position)>0):  #태풍 소멸 이후
            if min_position[last_key]['type'] == 'tc':
                print(pred_time.strftime("%Y/%m/%d/%HUTC"), "태풍이 소멸하였습니다.")
                min_position[last_key]['type'] = 'ex'
                
            
        elif (len(minima_positions) < 1) and (len(min_position)<1): #태풍 발생을 못 찾음
            print(pred_time.strftime("%Y/%m/%d/%HUTC"), "태풍이 발생X")
            pass
        
        
        elif (len(minima_positions) > 0) and (len(min_position)>0):
            dis_pos_list=[]

            if min_position[last_key]['type'] == 'tc':
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
            
            else:
                pass
                
        elif (len(minima_positions) > 0) and (len(min_position)<1):
            min_index = minima_positions[0]
            min_value = data_copy[int(minima_positions[0][0]),int(minima_positions[0][1])]
            min_lat = lat_indices[lat_start + min_index[0]]
            min_lon = lon_indices[lon_start + min_index[1]]


            min_position[pred_time] = {'lon': min_lon, 'lat': min_lat, 'idx': min_index, 
                                                                    'mslp':  min_value, 'type':'tc'}

    return min_position


#%%
storm_name = 'hinnamnor'                                                                               
storm_name = storm_name.upper()
storm_year = 2022

lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45, part='n') 
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

time_interval_list = np.arange(0,24*4,6)
first_str = '2022/08/27/00UTC' 
first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
datetime_list = np.array([(first_time + timedelta(hours=int(hours))).strftime("%Y/%m/%d/%HUTC") for hours in time_interval_list])
datetime_array = np.array([(first_time + timedelta(hours=int(hours))) for hours in time_interval_list])

storm_lon, storm_lat, storm_mslp, storm_time = storm_info(storm_name, storm_year, datetime_list = datetime_array, wind_thres=0)   #태풍 영문명, 년도 입력

min_position={}

for pred_str, pred_time in zip(datetime_list, datetime_array):
    input_data_dir = rf'{pangu_dir}/input_data/{pred_str}'
    met = Met(input_data_dir, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    
    mslp = met.met_data('MSLP')
    wind_speed = met.wind_speed()
    z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
    min_position = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                    wind_speed, pred_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                    min_position, mask_size = 2.5, init_size=2.5, local_min_size = 5, mslp_z_dis = 250)
    


mp_time = list(min_position.keys())
mp_time.sort()
past_reverse_time = datetime_array[datetime_array < mp_time[0]][::-1]

for past_time in past_reverse_time:
    past_str = past_time.strftime("%Y/%m/%d/%HUTC")
    input_data_dir = rf'{pangu_dir}/input_data/{past_str}'
    # surface_dir = rf'{pangu_dir}/input_data/{pred_str}/surface.npy'
    met = Met(input_data_dir, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    mslp = met.met_data('MSLP')
    wind_speed = met.wind_speed()
    z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
    min_position = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                        wind_speed, past_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                        min_position, mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 250)

min_position = {k: min_position[k] for k in sorted(min_position)}
#%%


fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.set_title(f'{first_str} \n{datetime_list[-1]}', fontsize=20, loc = 'left')
ax.set_title(f'ERA5 Track\n{storm_name}', fontsize=20, loc = 'right')
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



lons = [pos['lon'] for _,pos in min_position.items()]
lats = [pos['lat'] for _,pos in min_position.items()]
min_values = [pos['mslp'] for _,pos in min_position.items()]
pred_times = [pos for pos,_ in min_position.items()]

lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)

for i in range(len(pred_times)):
    if pred_times[i].hour == 0:
        ax.text(lons[i],lats[i], str(pred_times[i].day)
            , horizontalalignment='center', verticalalignment='bottom', fontsize=10)


    
ax.legend(loc='upper right')
plt.show()


plt.plot(pred_times, min_values, color = 'blue', label = 'ERA5')
plt.plot(storm_time, storm_mslp, color = 'black', label = 'Best track')
# Set the locator for the x-axis to locate days
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# Set the formatter for the x-axis to display dates in 'mm.dd' format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d'))

# Optionally, rotate the date labels for better readability
# plt.gcf().autofmt_xdate()
plt.ylabel('pressure(hPa)')
plt.xlabel('Time')
plt.legend(loc='lower left')
plt.show()


#%%
era_mean_data = {}

lon_space = 5
lat_space = 1
lon_space, lat_space = int(lon_space*4), int(lat_space*4)

variable_keys = list(upper_dict.keys())
variable_keys.append('ept')

for pred_time in min_position:
    print(pred_time)
    era_mean_data[pred_time]={}
    pred_str = pred_time.strftime("%Y/%m/%d/%HUTC")

    for key in variable_keys:
        era_mean_data[pred_time][key]=[]
        

        c_lat = min_position[pred_time]['idx'][0]
        c_lon = min_position[pred_time]['idx'][1]
            
        predict_interval = pred_time - first_time   
        predict_interval = int(predict_interval.total_seconds() / 3600)
            
        input_data_dir = rf'{pangu_dir}/input_data/{pred_str}'
        met = Met(input_data_dir, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
   
        if key != 'ept':
            variable = met.met_data(key, level = 'all')
        
        elif key == 'ept':
            t = met.met_data('t', level = 'all')
            q = met.met_data('q', level = 'all')
            variable = ep_t(t, pres_array[:,np.newaxis,np.newaxis], q)
        
        era_mean_data[pred_time][key] = variable[:, c_lat - lat_space: c_lat + lat_space+1, c_lon - lon_space: c_lon + lon_space+1]
        era_mean_data[pred_time][key] = np.mean(np.array(era_mean_data[pred_time][key]), axis=1)[::,:]

#%%
variable_dict = {
    't'  : {'title': 'Temperature anomaly', 'level': np.arange(200,300,5), 'diff_level': np.arange(-5,5.5,1), 
            'cmap': 'gist_ncar', 'bar_label': 'K'}, 
    'q'  : {'title': 'Specific Humidity', 'level': np.arange(0,20,0.5), 'diff_level': np.arange(-10,10.5,1), 
            'cmap': 'gist_ncar', 'bar_label': 'g/kg'}, 
    'ept': {'title': 'Equivalent Potential Temperature', 'level': np.arange(335,366,1), 'diff_level': np.arange(-20,20.5,1), 
            'cmap': 'gist_ncar', 'bar_label': 'K'},
    'z':   {'title': 'Geopotential Height anomaly', 'level': np.arange(0,21001,3000), 'diff_level': np.arange(-200,200.1,10), 
            'cmap': 'seismic', 'bar_label': 'm'},
    'u'  : {'title': 'Zonal Wind(U)', 'level': np.arange(-50,51,1), 'diff_level': np.arange(-50,51,1),
            'cmap': 'seismic', 'bar_label': 'm/s'}, 
    'v'  : {'title': 'Meriodonal Wind(V)', 'level': np.arange(-50,51,1), 'diff_level': np.arange(-50,50.1,1),
            'cmap': 'seismic', 'bar_label': 'm/s'} 
    }

x = np.arange(-lon_space/4,lon_space/4+0.1, 0.25)
y = pres_array

xx, yy = np.meshgrid(x, y)
# path = f'/Data/home/jjoo0727/Pangu-Weather/plot/ENS/group_all_level_lon_mean/{perturation_scale}ENS{surface_str}{upper_str}'
for pred_time in min_position:
    pred_str = pred_time.strftime("%Y.%m.%d.%HUTC")
    # print(era_mean_data[pred_time][0]['z'])

    # for plot_num in range(2):    
    fig, axs = plt.subplots(2,3,figsize = (20,12))
    fig.suptitle(pred_str, y=0.95, fontsize = 25)
        
    key_list = list(variable_dict.keys())

    for i, key in enumerate(key_list):
        ax = axs.ravel()[i]
        if key == 'z' or key == 't':
            cax = ax.contourf(xx, yy, era_mean_data[pred_time][key]-np.mean(era_mean_data[pred_time][key], axis=1)[:,np.newaxis], cmap = 'seismic', levels = variable_dict[key]['diff_level'], extend = 'both')
        else:
            cax = ax.contourf(xx, yy, era_mean_data[pred_time][key], cmap = variable_dict[key]['cmap'], levels = variable_dict[key]['level'], extend = 'both')
        ax.set_xlabel('Longitude(°)')
        ax.set_ylabel('Pressure(hPa)')
        ax.set_title(f'{variable_dict[key]["title"]}', fontsize = 14)
        ax.axhline(200, color='black', linestyle='--')
        ax.invert_yaxis()
        cbar= plt.colorbar(cax, ax = ax, shrink = 1)
        cbar.set_label(variable_dict[key]['bar_label'], fontsize=15)
        
    plt.subplots_adjust(wspace = 0.3, hspace = 0.4)
    
    path = f'/Data/home/jjoo0727/Pangu-Weather/plot/typhoon/{storm_year}_{storm_name}/lat_mean'
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    else:
        pass  
    # plt.show()
    
    plt.savefig(f'{path}/{pred_str}.png')
    plt.close()
#%%
file_folder = 'ERA5_z500,u500,850'

lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45, part='n')  
if file_folder.endswith('small'):
    lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(130,160,15,35, part='n')  
    # lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(120,140,25,45, part='n')  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

for pred_str, pred_time in zip(datetime_list, datetime_array):
    input_data_dir = rf'{pangu_dir}/input_data/{pred_str}'
    # surface_dir = rf'{pangu_dir}/input_data/{pred_str}/surface.npy'
    met = Met(input_data_dir, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    
    z_500 = met.met_data('z', level = 500)
    u_850 = met.met_data('u', level = 850)
    u_500 = met.met_data('u', level = 500)
    u_925 = met.met_data('u', level = 925)
    v_850 = met.met_data('v', level = 850)
    Temp = met.met_data('T2M')
    w_850 = met.wind_speed(level = 850)
    w_200 = met.wind_speed(level = 200)
    
    # mask = w_850 > 12
    # u_850 = np.where(mask, u_850, np.nan)  
    # v_850 = np.where(mask, v_850, np.nan) 
    
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{pred_str}', fontsize=20, loc = 'left')
    ax.set_extent(extent, crs=proj)
    setup_map(ax, back_color='n')
    
    cz = ax.contour(lon_grid, lat_grid, z_500, colors = 'black', levels = np.arange(6000-60*10,6001,60))
    # cz = ax.contour(lon_grid, lat_grid, u_500, colors = 'black', levels = np.arange(-50,51,5))
    # cz = ax.contour(lon_grid, lat_grid, u_925, colors = 'black', levels = np.arange(-30,31,3))
    plt.clabel(cz, fontsize=12)
    cw = ax.contourf(lon_grid, lat_grid, u_500, cmap = 'seismic', levels = np.arange(-30,31,3), extend = 'both')
    # cw = ax.contourf(lon_grid, lat_grid, Temp, cmap = 'jet', levels = np.arange(288,308,1), extend = 'both')
    # cw = ax.contourf(lon_grid, lat_grid, w_200, cmap = 'jet', levels = np.arange(30,61,10), extend = 'max')
    cbar = plt.colorbar(cw, ax = ax, shrink = 0.6, ticks = np.arange(-30,31,10))
    cbar.set_label('m/s', fontsize=15)
    
    step = 8
    if file_folder.endswith('small'):
        step = 4
    barb = ax.barbs(lon_grid[::step,::step], lat_grid[::step,::step], u_850[::step,::step], v_850[::step,::step], length=6)
    
    if storm_time[0] <= pred_time:
        slon = storm_lon[storm_time<=pred_time]
        slat = storm_lat[storm_time<=pred_time]
        mslp = storm_mslp[storm_time<=pred_time]
        
        plt.plot(slon, slat,marker=tcmarkers.TS, color = 'yellow',markersize = 10 ,transform = ccrs.PlateCarree())

    file_path = f'/Data/home/jjoo0727/Pangu-Weather/plot/typhoon/{storm_year}_{storm_name}/{file_folder}'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    else:
        pass  
    plt.savefig(f'{file_path}/{pred_str.replace("/", ".")}.png')
    plt.close()