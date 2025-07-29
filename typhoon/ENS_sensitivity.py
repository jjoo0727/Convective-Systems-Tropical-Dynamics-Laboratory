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
import matplotlib.ticker as ticker
import tcmarkers

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from skimage.measure import regionprops
from sklearn.decomposition import PCA

import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, minimum_filter, maximum_filter, label
from scipy import integrate
from scipy.sparse import diags, kron
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from datetime import datetime, timedelta

# import haversine
from haversine import haversine

import tropycal.tracks as tracks

from numba import jit

import itertools    

# from ty_pkg import latlon
from ty_pkg import truncate_colormap, colorline, setup_map, weather_map_contour, contourf_and_save, ep_t, concentric_circles, interpolate_data
from ty_pkg import latlon_extent, storm_info, haversine_distance, Met, calculate_bearing_position, tc_finder

pangu_dir = r'/home1/jek/Pangu-Weather'

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

# Define the colors you want in your colormap
colors = ["purple", "darkblue", "lightblue", "white", "yellow", "red", "pink"]

# Create a colormap from the colors
pwp = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# %%
#위경도 지정
lat_indices, lat_start, lat_end, lon_indices, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])


ssv_dict = {}


#태풍 지정
storm_name = 'hinnamnor'                                                                               
storm_name = storm_name.upper()
storm_year = 2022

surface_factors = []  # 예시: 지표면에서는 'MSLP'만 선택
upper_factors = ['z'] 
perturation_scale = 0.05

#예측 시간 지정, 초기 시간 지정, 앙상블 수
predict_interval_list = np.arange(0,24*7+1,6)  
ens_num = 100


for first_str in ['2022/08/27/00UTC']:
    ssv_key = first_str.split('/')[2]+'.'+first_str.split('/')[3][:2]
    surface_factors.sort()
    upper_factors.sort()
    surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
    upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가


    first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
    datetime_list = np.array([first_time + timedelta(hours=int(hours)) for hours in predict_interval_list])
    # datetime_array = np.array([(first_time + timedelta(hours=int(hours))) for hours in predict_interval_list])

    storm_lon, storm_lat, storm_mslp, storm_time = storm_info(pangu_dir, storm_name, storm_year, datetime_list = datetime_list, wind_thres=0)   #태풍 영문명, 년도 입력

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
                                    min_position[ens], mask_size = 2.5, init_size=5, local_min_size = 5, mslp_z_dis = 1000)


        for predict_interval in predict_interval_list[::-1]:
            predict_time = first_time + timedelta(hours=int(predict_interval))
            predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
            met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
            mslp = met.met_data('MSLP')
            wind_speed = met.wind_speed()
            z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
            
            min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                        wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                        min_position[ens], mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 1000)
            
            min_position[ens] = {k: min_position[ens][k] for k in sorted(min_position[ens])}
            
    ssv_dict[ssv_key] = min_position


for key_time ,min_position in ssv_dict.items():
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{key_time} (+{predict_interval_list[-1]}h)', fontsize=20, loc = 'left')
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
    # for ens in range(34,35):
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


    levels = np.linspace(0.0005, 0.015, 100)
    # cf = ax.contourf(lon_grid, lat_grid, f, levels=levels, transform=proj, cmap='jet')
    plt.show()

#%%
storm_lon
storm_time
storm_lat
storm_dict = {i: {'lon': x, 'lat': y} for i,x,y in zip(storm_time, storm_lon, storm_lat)}
storm_dict
#%%
ssv_dict['27.00'][0]
#%%
dis_dict = {}
key_time = datetime(2022,8,28, 12)
for e, data_total in ssv_dict['27.00'].items():
    
    for key_time, data in data_total.items():
        time_delta = (key_time - datetime(2022,8,27,0)).total_seconds() / 3600
        if time_delta not in dis_dict:
            dis_dict[time_delta] = []
        lon = data['lon']
        lat = data['lat']
        if key_time not in storm_dict:
            continue
        ty_lon, ty_lat = storm_dict[key_time]['lon'], storm_dict[key_time]['lat']
        d = haversine_distance(lat, lon, ty_lat, ty_lon)
        dis_dict[time_delta].append(d)
        
#%%
dis_dict
mean_dis_dict = {key: np.mean(values) for key, values in dis_dict.items()}
mean_dis_dict = {key: value for key, value in mean_dis_dict.items() if not np.isnan(value)}
#%%     

#s2s와 비교하기
import pickle  
import seaborn as sns 
with open('/home1/jek/Pangu-Weather/input_data/TIGGE/dis_error.pkl', 'rb') as tf:
    dis_error = pickle.load(tf)
dis_error

sns.set(style="whitegrid")

# Create a single figure to host all boxplots
plt.figure(figsize=(12, 8))

# Data needs to be restructured for seaborn boxplot, using a DataFrame
data = []
for t_delta, distances in dis_error.items():
    for distance in distances:
        data.append({'Time Delta': t_delta, 'Distance': distance})
df = pd.DataFrame(data)
#%%
# Create a boxplot
sns.boxplot(x='Time Delta', y='Distance', data=df)
plt.title('Distance Distribution by Time Delta')
# 카테고리 위치 계산
mean_dis_dict = {k: v for k, v in mean_dis_dict.items() if k in df['Time Delta'].unique()}
categories = df['Time Delta'].unique()
x_positions = [list(categories).index(time_delta) for time_delta in mean_dis_dict.keys()]
mean_distances = list(mean_dis_dict.values())

# 산점도 그리기
# plt.scatter(x_positions, mean_distances, color='green', marker='x', s=100, zorder = 12)
# dis_dict의 모든 거리를 산점도로 추가
dis_dict = {k: v for k, v in dis_dict.items() if k in df['Time Delta'].unique()}
# data_list = []
for time_delta, distances in dis_dict.items():

    # 각 Time Delta에 대해 박스플롯의 x 위치 찾기
    categories = df['Time Delta'].unique()  # 박스플롯의 x축 카테고리
    x_position = list(categories).index(time_delta)  # 현재 time_delta의 인덱스 위치

    # x_positions 리스트를 distances의 길이만큼 반복
    x_positions = [x_position] * len(distances)

    # 산점도 추가
    plt.scatter(x_positions, distances, color='red', marker='x', s=30, zorder = 10)

plt.xlabel('Time Delta (hours)')
plt.ylabel('Distance (km)')

# Show the plot
plt.show()

#%%
df
#%%
#각 앙상블 멤버 기압 변화 그리기
#34번 멤버가 이상함 
   
plt.figure(figsize=(10, 5))  # 그래프 크기 설정
for key_time ,min_position in ssv_dict.items():
    for ens, mp in min_position.items():
        
        # 데이터 추출
        dates = list(mp.keys())
        mslp_values = [mp[date]['mslp'] for date in dates]

        # 그래프 그리기
        plt.plot(dates, mslp_values, marker='o', linestyle='-')  # 선 그래프 그리기

# X축을 날짜 형식으로 설정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# 그래프 제목 및 레이블 설정
plt.title('MSLP Over Time')
plt.xlabel('Date')
plt.ylabel('MSLP (hPa)')

# 그래프 레이아웃 조정 및 표시
# plt.gcf().autofmt_xdate()  # 날짜 레이블이 서로 겹치지 않도록 조정
plt.grid(True)  # 그리드 표시
plt.show()
#%% 
# 없는 ens 번호들 찾기

missing_ens = []
date_key = datetime(2022, 8, 28, 0, 0)
# 모든 앙상블 멤버에 대해 반복
for ens in range(ens_num):
    # 날짜 객체 생성
    

    # 특정 날짜의 데이터가 있는지 확인
    if date_key not in ssv_dict['27.00'][ens]:
        missing_ens.append(ens)  # 없으면 리스트에 추가

# 출력
if missing_ens:
    print(f"Missing date data for ensemble members: {missing_ens}")
else:
    print("All ensemble members have data for the specified date.")


#%%

class WindFieldSolver:
    
    #######################################################################
    # Get TY wind field solving Poisson eq
    # Presume source term from average relative vort, divergence in TY area
    #######################################################################
    def __init__(self, lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850, radius=333, dx=111e3/4, dy=111e3/4):
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
            dis_dict[i] = haversine_distance(self.center_lat, self.center_lon, lat_grid[v_pos[0], v_pos[1]], lon_grid[v_pos[0], v_pos[1]])
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
        # Create arrays for center latitude and longitude
        center_lat_array = np.ones_like(self.lat_grid) * self.center_lat
        center_lon_array = np.ones_like(self.lon_grid) * self.center_lon

        # Calculate distances using the Haversine formula
        distances = haversine_distance(center_lat_array, center_lon_array, self.lat_grid, self.lon_grid)

        # Define the maximum distance to calculate intervals dynamically
        max_distance = 800  # Adjust this value as needed
        interval_size = 50
        intervals = [(i, i + interval_size) for i in range(0, max_distance, interval_size)]
        
        previous_average = None
        optimal_radius = None
        vort_list = []
        interval_sign = 0
        # Calculate average values for each interval and check for increasing pattern
        for i, (low, high) in enumerate(intervals):
            mask = (distances > low) & (distances <= high)
            if np.any(mask):
                masked_sf = self.vort_850[mask]
                
                current_average = np.mean(masked_sf)
                vort_list.append([high, np.mean(self.vort[mask]), np.std(self.vort[mask])])
                
                # Check if the average is increasing
                if previous_average is not None and current_average > previous_average and interval_sign == 0:
                    optimal_radius = intervals[i-1][1]  # The upper bound of the previous interval
                    interval_sign +=1
                    # break
                previous_average = current_average

        # If no increase is found, consider the last interval's upper bound as optimal radius
        if optimal_radius is None:
            optimal_radius = intervals[-1][1]

        # Create final mask based on optimal radius and apply to data
        optimal_radius = 350
        final_mask = distances <= optimal_radius
        sf_data = np.zeros_like(self.vort)
        vp_data = np.zeros_like(self.div)
        sf_data[final_mask] = self.vort[final_mask]
        vp_data[final_mask] = self.div[final_mask]
        # print(optimal_radius)
        vort_list = np.array(vort_list)
        return sf_data, vp_data, optimal_radius, vort_list
        
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
        sf_source, vp_source, optimal_radius, vort_list = self.create_source_term()
        sf = self.solve_poisson_with_scaling(self.lat_grid, sf_source, self.dx, self.dy, scaling_factors)
        vp = self.solve_poisson_with_scaling(self.lat_grid, vp_source, self.dx, self.dy, scaling_factors)
        u_sf, v_sf = self.compute_wind_field(sf, 'sf')
        u_vp, v_vp = self.compute_wind_field(vp, 'vp')
        u = u_sf + u_vp
        v = v_sf + v_vp
        # return u, v, u_sf, v_sf, u_vp, v_vp
        # return u, v, sf_source, vort_list, optimal_radius, self.center_lat, self.center_lon
        return u, v


#%%
# 예제 데이터 (ssv_dict와 ens_num, z_500 데이터가 이미 정의되어 있다고 가정)
key1 = '27.00'
# steer_pres = [250]
steer_pres = [850,700,600,500,400,300,250]
altitude = 500
predict_interval = 120

ens_num_list = []
# for datetime1 in [datetime(2022,8,28,0,0),datetime(2022,8,29,0,0),datetime(2022,8,30,0,0),datetime(2022,8,31,0,0), datetime(2022,9,1,0,0)]:
for datetime1 in [datetime(2022,8,28,0,0)]:
    predict_interval = datetime1-datetime(2022, 8, 27, 0, 0)
    predict_interval = int(predict_interval.total_seconds() / 3600)
    
    
    for ens in range(ens_num):
        if datetime(2022,9,1,0,0) in ssv_dict[key1][ens]:
            ens_num_list.append(ens)
    print(ens_num_list)
    


    # 데이터를 추출합니다
    ens_pos = [(ens, ssv_dict[key1][ens][datetime(2022,9,1,0,0)]['lon'], ssv_dict[key1][ens][datetime(2022,9,1,0,0)]['lat']) for ens in ens_num_list]

    # 위도(lat) 기준으로 데이터를 정렬합니다
    pos_sorted_by_lat = sorted(ens_pos, key=lambda x: x[2])  # x[2]는 위도를 나타냅니다

    # 위도가 가장 낮은 10개와 가장 높은 10개를 추출합니다
    group1 = pos_sorted_by_lat[:10]  # 가장 낮은 10개
    group2 = pos_sorted_by_lat[-10:]  # 가장 높은 10개
    group3 = pos_sorted_by_lat[len(pos_sorted_by_lat) // 2 - 5:len(pos_sorted_by_lat) // 2 + 5]

    # group1과 group2에 있는 ens 번호만 추출합니다
    group1 = [item[0] for item in group1]
    group2 = [item[0] for item in group2]
    group3 = [item[0] for item in group3]
    tc_pos = [(ssv_dict[key1][ens][datetime1]['lon'], ssv_dict[key1][ens][datetime1]['lat']) for ens in ens_num_list]
    pc_pos = [(ssv_dict[key1][ens][datetime(2022,9,1,0,0)]['lon'], ssv_dict[key1][ens][datetime(2022,9,1,0,0)]['lat']) for ens in ens_num_list]
    
    # NumPy 배열로 변환
    tc_pos, pc_pos = np.array(tc_pos), np.array(pc_pos)
    # 경도의 왜곡을 보정
    corr_pos = np.copy(pc_pos)
    corr_pos[:, 0] = (pc_pos[:, 0]-np.mean(pc_pos[:, 0])) * np.cos(np.radians(pc_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca = PCA(n_components=1)
    pca.fit(corr_pos)
    pca.mean_[0] = pca.mean_[0] / np.cos(np.radians(pca.mean_[1])) + np.mean(pc_pos[:, 0])
    # print(pca.mean_)
    # plt.scatter(corr_pos[:,0],corr_pos[:,1])
    # plt.scatter(pca.mean_[0], pca.mean_[1])


    # 데이터를 주축에 투영
    projection = pca.transform(corr_pos)[:, 0]  # 주축에 투영된 데이터 (1차원)
    principal_component = pca.components_[0]

    # 투영된 데이터의 ensemble mean 계산
    ensemble_mean = np.mean(projection)

    # 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산
    distances = projection - ensemble_mean

    # 각 앙상블 멤버의 거리를 저장
    ensemble_distances = {ens: distance for ens, distance in enumerate(distances)}


    for choosen_factor in ['steering_wind']:

        ens_factor=[]
        u_mean = []
        v_mean = []
        
        
        for ens in ens_num_list:
            center_lon, center_lat = ssv_dict[key1][ens][datetime1]['lon'], ssv_dict[key1][ens][datetime1]['lat']
            
            output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
            met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
            
            if choosen_factor != 'steering_wind':
                ens_factor.append(met.met_data(choosen_factor, level = altitude))
            else:
                u_list = []
                v_list = []
                
                
                if len(steer_pres) > 1:
                    for steer_altitude in steer_pres:
                        
                        div = met.divergence(level = steer_altitude)
                        vort = met.vorticity(level = steer_altitude)
                        vort = met.vorticity(level = 850)
                        ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850)
                        u_ty, v_ty = ty_wind.solve()
                        
                        
                        u_list.append(met.met_data('u', level = steer_altitude)-u_ty)
                        v_list.append(met.met_data('v', level = steer_altitude)-v_ty)
                    
                    u,v = np.zeros(np.shape(u_list[0])), np.zeros(np.shape(u_list[0]))
                   
                    for i in range(len(steer_pres)-1):
                        u += (u_list[i]+u_list[i+1])/2*(steer_pres[i]-steer_pres[i+1])
                        v += (v_list[i]+v_list[i+1])/2*(steer_pres[i]-steer_pres[i+1])
                    
                    u/=np.ptp(steer_pres)
                    v/=np.ptp(steer_pres)
                    
                else:
                    div = met.divergence(level = steer_pres[0])
                    vort = met.vorticity(level = steer_pres[0])
                    ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850)
                    u_ty, v_ty = ty_wind.solve()
                    u = met.met_data('u', level = steer_pres[0]-u_ty)
                    v = met.met_data('v', level = steer_pres[0]-v_ty)
                
                u_mean.append(u)
                v_mean.append(v)
                ens_factor.append(u * principal_component[0] + v * principal_component[1])
                
                if ens == 0:
                    u0 = u
                    v0 = v
                    
        u_mean = np.mean(np.array(u_mean), axis=0)
        v_mean = np.mean(np.array(v_mean), axis=0)
            
        # z_500 데이터를 numpy 배열로 변환
        ens_factor = np.array(ens_factor)
        # print(np.shape(ens_factor))
        
        ens_factor_std = np.std(ens_factor, axis=0)
        ens_factor = (ens_factor - np.mean(ens_factor, axis=0)) / ens_factor_std

        # 공분산 및 분산 계산
        cov_matrix = np.zeros_like(ens_factor[0])
        var_matrix = np.zeros_like(ens_factor[0])

        for i in range(ens_factor.shape[1]):  # lat 방향
            for j in range(ens_factor.shape[2]):  # lon 방향
                cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
                var_matrix[i, j] = np.var(ens_factor[:,i,j])


        # 공분산/분산 비율 계산
        cov_var_ratio = cov_matrix / var_matrix * 111

        # NaN 또는 Inf 값을 0으로 대체
        cov_var_ratio = np.nan_to_num(cov_var_ratio)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        contour = ax.contourf(lon_grid, lat_grid, cov_var_ratio, cmap=pwp, levels=np.linspace(-400, 400, 21), transform=ccrs.PlateCarree())
        cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}_{altitude}) / {choosen_factor}_{altitude}', shrink = 0.7)
        cbar.locator = ticker.MultipleLocator(100)  # Set the colorbar ticks to have an interval of 0.5
        cbar.update_ticks()
        ax.scatter(tc_pos[:, 0], tc_pos[:, 1], alpha=0.7, s=0.6, c='r')
        setup_map(ax, back_color='n')
        # ax.set_extent([100,160,5,45])
        # plt.plot(np.mean(tc_loc, axis=0)[0], np.mean(tc_loc, axis=0)[1], marker = tcmarkers.TS, color = 'red')
        plt.title(f'{storm_name}\n{choosen_factor.capitalize()} {altitude}hPa', loc = 'right')
        plt.title(f'08.{key1}UTC(+{predict_interval}h)\n{datetime1.strftime("%m.%d.%HUTC")}', loc = 'left')
        if choosen_factor == 'steering_wind':
            ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5)
            if len(steer_pres) > 1:
                plt.title(f'{storm_name}\n{choosen_factor.capitalize()} {steer_pres[0]}-{steer_pres[-1]}hPa', loc = 'right')
            else:
                plt.title(f'{storm_name}\n{choosen_factor.capitalize()} {steer_pres[0]}hPa', loc = 'right')
        
        
        mean_lon = pca.mean_[0]  # 경도 보정 복구
        # print(mean_lon)
        mean_lat = pca.mean_[1]
        principal_lon = pca.components_[0, 0] / np.cos(np.radians(pca.mean_[1]))  # 경도 방향성분 보정 복구
        principal_lat = pca.components_[0, 1]        
        # ax.quiver(pca.mean_[0], pca.mean_[1], pca.components_[0, 0], pca.components_[0, 1], scale=20, color='r', width=0.003, label='Principal Axis')
        ax.quiver(mean_lon, mean_lat, principal_lon, principal_lat, scale=20, color='r', width=0.003, label='Principal Axis')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()


#%%
forecast_hour = 144
steer_altitude = 500

for ens_num in [2]:
    # for steer_altitude in [1000, 850, 500, 200]:
    for steer_altitude in [850]:
        for forecast_hour in np.arange(24,169,24):
        # for forecast_hour in [168]:
            output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens_num}'
            met = Met(output_data_dir, forecast_hour, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
            div = met.divergence(level = steer_altitude)
            vort = met.vorticity(level = steer_altitude)
            vort_850 = met.vorticity(level = 850)
            u = met.met_data('u', level = steer_altitude)
            z = met.met_data('z', level = steer_altitude)
            v = met.met_data('v', level = steer_altitude)


            key_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")+timedelta(hours=int(forecast_hour))
            center_lon = ssv_dict['27.00'][ens_num][key_time]['lon']
            center_lat = ssv_dict['27.00'][ens_num][key_time]['lat']

            # fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            # setup_map(ax, back_color='n')
            # sax = ax.scatter(lon_grid, lat_grid, c=div, cmap = 'seismic', vmin = -1e-4, vmax = 1e-4)
            # cbar = plt.colorbar(sax, orientation = 'horizontal', shrink = 0.8, pad = 0.05)
            # cbar.set_label('$s^{-1}$')
            # plt.show()
            # fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            # plt.title(f"{ens_num}ENS (+{forecast_hour}h) {steer_altitude}hPa", fontsize = 15)
            # setup_map(ax, back_color='n')
            # cax = ax.contour(lon_grid, lat_grid, z, levels = np.arange(1200,1621,30), zorder = 20, colors = 'black')
            # plt.clabel(cax)
            # plt.plot(center_lon, center_lat, marker = tcmarkers.TS, color = 'black', markersize=5, label = 'MSLP Min')

            # solver = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850, radius = 125)
            # u_ty, v_ty, sf_source, vort_list, optimal_radius,center_lat, center_lon = solver.solve()

            # sax = ax.scatter(lon_grid, lat_grid, c=vort, cmap = 'seismic', vmin = -1e-3, vmax = 1e-3)
            # cbar = plt.colorbar(sax, orientation = 'horizontal', shrink = 0.8, pad = 0.05)
            # cbar.set_label('$s^{-1}$')
            # plt.plot(center_lon, center_lat, marker = tcmarkers.TS, color = 'yellow', markersize=5, label = 'Vort Max')
            # plt.legend()
            # plt.show()

            solver = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850, radius = 125)
            u_ty, v_ty, sf_source, vort_list, optimal_radius,center_lat, center_lon = solver.solve()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            setup_map(ax, back_color='n')
            sax = ax.scatter(lon_grid, lat_grid, c=sf_source, cmap = 'seismic', vmin = -1e-4, vmax = 1e-4)
            cbar = plt.colorbar(sax, orientation = 'horizontal', shrink = 0.8, pad = 0.05)
            cbar.set_label('$s^{-1}$')
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            setup_map(ax, back_color='n')
            ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], u[::4,::4], v[::4,::4])
            plt.show()
            
            u -= u_ty
            v -= v_ty
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            setup_map(ax, back_color='n')
            ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], u[::4,::4], v[::4,::4])
            plt.show()



            # fig, ax = plt.subplots(1,3, figsize = (20,7))
            # fig.suptitle(f"{ens_num}ENS (+{forecast_hour}h) {steer_altitude}hPa R:{optimal_radius}km", fontsize = 20)

            # ax[0].set_title('Mean', fontsize=15)
            # ax[0].plot(vort_list[:,0],vort_list[:,1])
            # ax[0].axhline(y=0, color='r', linestyle='--')
            # # ax[0].axvline(x = optimal_radius, color='black')
            # ax[0].tick_params(labelsize = 15)
            # ax[0].set_xlabel('Radius(km)', fontsize=15)
            # ax[0].set_ylabel('Vorticity Mean($s^{-1}$)', fontsize=15)
            # ax[0].ticklabel_format(scilimits=(-3,3))        #지수 표현
            # ax[0].yaxis.get_offset_text().set_fontsize(15)  #지수 크기

            # ax[1].set_title('STD', fontsize=15)
            # ax[1].plot(vort_list[:,0],vort_list[:,2])
            # # ax[1].axvline(x = optimal_radius, color='black')
            # ax[1].set_xlabel('Radius(km)', fontsize=15)
            # ax[1].set_ylabel('Vorticity STD($s^{-1}$)', fontsize=15)
            # ax[1].ticklabel_format(scilimits=(-3,3))
            # ax[1].tick_params(labelsize = 15)
            # ax[1].yaxis.get_offset_text().set_fontsize(15)

            # ax[2].set_title('STD/Mean', fontsize=15)
            # ratio = vort_list[:,2]/vort_list[:,1]
            # ax[2].plot(vort_list[:,0],ratio)
            # ax[2].set_ylim(-5,5)
            # # ax[2].axvline(x = optimal_radius, color='black')
            # ax[2].axhline(y=0.5, color='r', linestyle='--')
            # ax[2].axhline(y=1, color='r', linestyle='--')
            # ax[2].axhline(y=2, color='r', linestyle='--')
            # ax[2].set_xlabel('Radius(km)', fontsize=15)
            # ax[2].set_ylabel('STD/Mean', fontsize=15)
            # ax[2].tick_params(labelsize = 15)
            # plt.show()
            
# exceeds_one = ratio > 1
# print(exceeds_one)
# print(vort_list[exceeds_one, 0][0])
# print(ratio[exceeds_one][0])
# ax[2].axvline(x=vort_list[exceeds_one, 0][0], color='r', linestyle='--')
# ax[2].text(vort_list[exceeds_one, 0][0], -0.05, f'{int(vort_list[exceeds_one, 0][0])}', transform=ax[2].get_xaxis_transform(),
#                color='red', ha='center', va='top')

#%%
ssv_dict['27.00'][82][datetime(2022,9,2,0,0)]
#%%
for steer_altitude in steer_pres:
    print(steer_altitude)
    div = met.divergence(level = steer_altitude)
    vort = met.vorticity(level = steer_altitude)
    # vort = np.zeros_like(vort)
    # dib  = np.zeros_like(div)
    solver = WindFieldSolver(lat_grid, lon_grid, 26.75, 133.0, vort, div, radius=200)
    u_ty, v_ty = solver.solve()
    u, v = met.met_data('u', level = steer_altitude)-u_ty, met.met_data('v', level = steer_altitude)-v_ty
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], u[::4, ::4], v[::4, ::4])
    setup_map(ax, back_color='n')
    plt.show()
    

#%%
ssv_dict['27.00'][82][datetime(2022,8,29,0,0)]['lon']
#%%
# Usage
# Assuming lat_grid and lon_grid are defined properly
# solver = WindFieldSolver(lat_grid, lon_grid, 27, 133, vort, div, radius = 125)
solver = WindFieldSolver(lat_grid, lon_grid, 24.5, 129.75, vort, div, radius = 125)
# solver = WindFieldSolver(lat_grid, lon_grid, 26.75, 133, vort, div, radius = 200)
# u, v, u_sf, v_sf, u_vp, v_vp = solver.solve()
u_ty, v_ty, sf_source = solver.solve()


fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# fig, ax = plt.subplots(figsize=(10, 5))
setup_map(ax, back_color='n')
strm = ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], u[::4,::4], v[::4,::4], scale=500)
# strm = ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], u_vp[::4,::4], v_vp[::4,::4], scale=50)
# strm = ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], u_sf[::4,::4], v_sf[::4,::4], scale=250)
ax.quiverkey(strm, X=0.9, Y=1.05, U=10, label='10m/s', labelpos='E')
ax.set_title('Wind Field Visualization')
ax.set_xlabel('Longitude (degrees)')
ax.set_ylabel('Latitude (degrees)')
plt.show()
#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
setup_map(ax, back_color='n')
ax.scatter(lon_grid, lat_grid, c=sf_source)
