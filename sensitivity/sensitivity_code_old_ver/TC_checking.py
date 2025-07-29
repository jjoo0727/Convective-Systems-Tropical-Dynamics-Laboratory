#%%
import os
from tokenize import group
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
import matplotlib.ticker as mticker
import tcmarkers

import pickle  

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
from ty_pkg import truncate_colormap, colorline, setup_map, weather_map_contour, contourf_and_save, ep_t, concentric_circles, interpolate_data, set_map
from ty_pkg import latlon_extent, storm_info, haversine_distance, Met, calculate_bearing_position, tc_finder, WindFieldSolver, find_large_groups

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


#%%
#! 태풍 경로 정보 기존 정보 불러오기
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
key_time_list = ['2022/08/27/00UTC']
predict_interval_list = np.arange(0,24*7+1,6)  
ens_list = range(0,4000)
new_ssv = 'n'           #새로 생성할 것인지 여부, n이면 기존 파일 불러옴
retro_opt = 'td'        #다시 돌아가면서 태풍 추적시 강한 것만 추적하려면 td로

if retro_opt =='td':
    retro_opt = '_td'
else:
    retro_opt = ''
        
#! 태풍 경로 정보 새로 생성하기
if new_ssv == 'y':
    for first_str in key_time_list:
        first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
        key_str = first_time.strftime("%m.%d.%HUTC")
        ssv_key = first_time
        surface_factors.sort()
        upper_factors.sort()
        surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
        upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가


        datetime_list = np.array([first_time + timedelta(hours=int(hours)) for hours in predict_interval_list])
        # datetime_array = np.array([(first_time + timedelta(hours=int(hours))) for hours in predict_interval_list])

        storm_lon, storm_lat, storm_mslp, storm_time = storm_info(pangu_dir, storm_name, storm_year, datetime_list = datetime_list, wind_thres=0)   #태풍 영문명, 년도 입력

        min_position = {}  # 태풍 중심 정보 dictionary



        # for ens in range(ens_num):
        for ens in ens_list:
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
              
                #mask_size는 다음 태풍 찾을 때 그 위경도 안이 아니면 안 찾음S
                #처음 시작할 때는 5도 이내에만 들어오면 되고
                #mslp_z_dis는 250이 기본인데 이는 mslp 지역 최솟값과 z지역 최댓값이 250km 이내여야 pass
                #mslp_2hpa는 주변 8방위로 2hPa이 감소하는지 보는건데 일단은 'n'으로 걸어둠
                #아래에서는 mask_size말고는 아무런 제약을 안 걸었음
                min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                        wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                        min_position[ens], mask_size = 2.5, init_size=5, local_min_size = 5, mslp_z_dis = 250, wind_thres=8)

            
            for predict_interval in predict_interval_list[::-1]:
                predict_time = first_time + timedelta(hours=int(predict_interval))
                predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
                met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                mslp = met.met_data('MSLP')
                wind_speed = met.wind_speed()
                z_diff = met.met_data('z', level = 300) - met.met_data('z', level = 500)
                
                if retro_opt != '_td':
                    min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                                wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                                min_position[ens], mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 1000, wind_thres=8)
                else:
                    min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                                wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                                min_position[ens], mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 250, wind_thres=8)
                    
                
                min_position[ens] = {k: min_position[ens][k] for k in sorted(min_position[ens])}
                
        ssv_dict[ssv_key] = min_position

    with open(rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict{retro_opt}_{min(ens_list)}_{max(ens_list)}.pkl', 'wb') as f:
        pickle.dump(ssv_dict, f)

else:
    for first_str in key_time_list:
        first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
        key_str = first_time.strftime("%m.%d.%HUTC")
        ssv_key = first_time
        surface_factors.sort()
        upper_factors.sort()
        surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
        upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가
        
        datetime_list = np.array([first_time + timedelta(hours=int(hours)) for hours in predict_interval_list])
        # datetime_array = np.array([(first_time + timedelta(hours=int(hours))) for hours in predict_interval_list])
        storm_lon, storm_lat, storm_mslp, storm_time = storm_info(pangu_dir, storm_name, storm_year, datetime_list = datetime_list, wind_thres=0)   #태풍 영문명, 년도 입력

        
        with open(rf'/data03/Pangu_TC_ENS/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict{retro_opt}_{min(ens_list)}_{max(ens_list)}.pkl', 'rb') as f:
            ssv_dict = pickle.load(f)
        

#%%
#! 가장 상관관계 높은 axis 구하기, 2nd main code
# 시간 지정
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)         #분석 시작 시점
target_time = datetime(2022,9,1,0)          #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")

total_time_range = int((target_time - key_time).total_seconds() / 3600)
start_time_range = int((start_time  - key_time).total_seconds() / 3600)

# 변수 지정
nearby_sign = 'n'                           #가까운 태풍만 추출
distance_threshold = 0                      #가까운 태풍의 거리
steering_sign = 'y'                         #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
steer_uni_alt = 0                           #steering wind를 구할 때, 고도를 하나
# choosen_factor_list = ['z','t','q']       #구하고자 하는 변수
choosen_factor_list = ['z']                 #구하고자 하는 변수
# altitude_list = [1000,850,700,500,300,200]#각 변수에 대해 구하고자 하는 고도
altitude_list = [850,500,250]               #각 변수에 대해 구하고자 하는 고도
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x
axis_opt = 'quiver'                         #axis 뭘로 잡을지, opt: 위치 상관관계 최대인 axis, tar: 최종 위치의 axis, mid: 중간 위치의 axis, lon: 경도, lat: 위도
data_sign = 'y'                             #기존의 데이터를 사용할 것인지, n이면 새로 구함
predict_interval = 72

if nearby_sign == 'y':
    nearby_sign_name = '_nearby'
else:
    nearby_sign_name = ''
    

# steering_sign이 y면 steering wind에 대해서만 구하기
if steering_sign == 'y':
    altitude_list = ['850_200']
    choosen_factor_list = ['steering_wind']
    

ens_num_list = []
ens_not_list = []

for ens in ens_list:
    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
        ens_num_list.append(ens)
    else:
        ens_not_list.append(ens)

print(len(ens_num_list), len(ens_not_list))




datetime1 = key_time + timedelta(hours=int(predict_interval))

ens_num_list = []
for ens in ens_list:
    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
        ens_num_list.append(ens)




mid_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]

# NumPy 배열로 변환
mid_pos, tar_pos = np.array(mid_pos), np.array(tar_pos)

# 경도의 왜곡을 보정
corr_pos_tar = np.copy(tar_pos)
corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
pca_tar = PCA(n_components=1)
pca_tar.fit(corr_pos_tar)
pca_tar.mean_[0] = pca_tar.mean_[0] / np.cos(np.radians(pca_tar.mean_[1])) + np.mean(tar_pos[:, 0])


if nearby_sign == 'y':
    tar_pos, mid_pos, group_idx = find_large_groups(mid_pos, ens_num_list, tar_pos, nearby_sign, distance_threshold, 1, 10)
    print(len(ens_num_list), len(group_idx))
    
corr_pos_mid = np.copy(mid_pos)
corr_pos_mid[:, 0] = (mid_pos[:, 0]-np.mean(mid_pos[:, 0])) * np.cos(np.radians(mid_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
pca_mid = PCA(n_components=1)
pca_mid.fit(corr_pos_mid)
pca_mid.mean_[0] = pca_mid.mean_[0] / np.cos(np.radians(pca_mid.mean_[1])) + np.mean(mid_pos[:, 0])
projection = pca_mid.transform(corr_pos_mid)[:, 0]  # 주축에 투영된 데이터 (1차원)

#distance 구하기
corr_pos_tar = np.copy(tar_pos) #nearby를 고려하여 다시 target 부르기 PCA는 전체 데이터로 해야되므로 nearby 이전에 진행
corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
# projection = pca_tar.transform(corr_pos_tar)[:, 0]  # 주축에 투영된 데이터 (1차원)
principal_component = pca_tar.components_[0]

# 투영된 데이터의 ensemble mean 계산
ensemble_mean = np.mean(projection)

# 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산
distances = projection - ensemble_mean

# 각 앙상블 멤버의 거리를 저장
ensemble_distances = {ens: distance for ens, distance in zip(ens_num_list, distances)}

# 데이터를 추출합니다   
ens_pos = [(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]

# 거리리 기준으로 데이터를 정렬합니다
ens_pos_sorted = sorted(ens_pos, key=lambda x: ensemble_distances[x[0]])

# 거리 기준준 가장 낮은 10개와 가장 높은 10개를 추출합니다
group1 = ens_pos_sorted[:10]  # 가장 낮은 10개
group2 = ens_pos_sorted[-10:]  # 가장 높은 10개
group3 = ens_pos_sorted[len(ens_pos_sorted) // 2 - 5:len(ens_pos_sorted) // 2 + 5]

# group1과 group2에 있는 ens 번호만 추출합니다
group1 = [item[0] for item in group1]
group2 = [item[0] for item in group2]
group3 = [item[0] for item in group3]
group1_idx = np.array([ens_num_list.index(idx) for idx in group1])
group2_idx = np.array([ens_num_list.index(idx) for idx in group2])
group3_idx = np.array([ens_num_list.index(idx) for idx in group3])


mid_proj = pca_mid.transform(mid_pos)
tar_proj = pca_tar.transform(tar_pos)
mid2tar = pca_tar.transform(mid_pos)
mid_re = pca_mid.inverse_transform(mid_proj)
tar_re = pca_tar.inverse_transform(tar_proj)
tar2mid = pca_tar.inverse_transform(mid2tar)


group1_z_500 = []
for ens in group1:
    output_data_dir = rf'/data03/Pangu_TC_ENS/0.05ENS_z/{ens}'
    predict_time = first_time + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    mslp = met.met_data('MSLP')
    wind_speed = met.wind_speed()
    group1_z_500.append(met.met_data('z', level = 500))

group2_z_500 = []
for ens in group2:
    output_data_dir = rf'/data03/Pangu_TC_ENS/0.05ENS_z/{ens}'
    predict_time = first_time + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    mslp = met.met_data('MSLP')
    wind_speed = met.wind_speed()
    z_500 = met.met_data('z', level = 500)
    group2_z_500.append(z_500)

group1_z_500 = np.array(group1_z_500)
group2_z_500 = np.array(group2_z_500)

z_500_levels = np.arange(5580, 5941, 30)
# 첫 번째 플롯
fig, ax = plt.subplots(1, 1, figsize=(10 * latlon_ratio, 10), subplot_kw={'projection': proj})
ax.coastlines()  # 해안선 추가
plt.extent = [125, 160, 20, 40]
contourf_plot = ax.contourf(lon_grid, lat_grid, group1_z_500.mean(axis=0), cmap='jet', levels=z_500_levels, transform=ccrs.PlateCarree())
contour_plot = ax.contour(lon_grid, lat_grid, group1_z_500.mean(axis=0), colors='black', levels=z_500_levels, transform=ccrs.PlateCarree())
ax.clabel(contour_plot, inline=True, fontsize=10)
plt.colorbar(contourf_plot, ax=ax, orientation='vertical', label='Height (m)')
plt.title('Group 1: Z 500 Mean')
plt.show()

# 두 번째 플롯
fig, ax = plt.subplots(1, 1, figsize=(10 * latlon_ratio, 10), subplot_kw={'projection': proj})
ax.coastlines()  # 해안선 추가
plt.extent = [125, 160, 20, 40]
contourf_plot = ax.contourf(lon_grid, lat_grid, group2_z_500.mean(axis=0), cmap='jet', levels=z_500_levels, transform=ccrs.PlateCarree())
contour_plot = ax.contour(lon_grid, lat_grid, group2_z_500.mean(axis=0), colors='black', levels=z_500_levels, transform=ccrs.PlateCarree())
ax.clabel(contour_plot, inline=True, fontsize=10)
plt.colorbar(contourf_plot, ax=ax, orientation='vertical', label='Height (m)')
plt.title('Group 2: Z 500 Mean')
plt.show()

# 세 번째 플롯 (차이)
fig, ax = plt.subplots(1, 1, figsize=(10 * latlon_ratio, 10), subplot_kw={'projection': proj})
ax.coastlines()  # 해안선 추가
plt.extent = [125, 160, 20, 40]
diff_data = group1_z_500.mean(axis=0) - group2_z_500.mean(axis=0)
contourf_plot = ax.contourf(lon_grid, lat_grid, diff_data, cmap='jet', levels=np.arange(-50,51,5), transform=ccrs.PlateCarree())
contour_plot = ax.contour(lon_grid, lat_grid, diff_data, colors='black', levels=np.arange(-50,51,5), transform=ccrs.PlateCarree())
ax.clabel(contour_plot, inline=True, fontsize=10)
plt.colorbar(contourf_plot, ax=ax, orientation='vertical', label='Height Difference (m)')
plt.title('Difference: Group 1 - Group 2')
plt.show()

#%%
predict_interval = 72
group1_z_500.mean(axis=0)-group2_z_500.mean(axis=0)


base_output_path = os.path.join(
                pangu_dir, 
                'output_data', 
                'steering_wind',
                key_str, 
                f'{perturation_scale}ENS{surface_str}{upper_str}', 
                f'{min(ens_list)}_{max(ens_list)}_{start_str}_{target_str}'
            )

ens_factor_uv_path = os.path.join(base_output_path, 'ens_factor_uv')
total_remove_uv_path = os.path.join(base_output_path, 'total_remove_uv')
ens_factor_uv = np.load(f'{ens_factor_uv_path}/{predict_interval}h{retro_opt}.npy')
total_remove_uv = np.load(f'{total_remove_uv_path}/{predict_interval}h{retro_opt}.npy')

#%%
#그룹 간 quiver 및 streamline 그리기기
group1_u = ens_factor_uv[group1_idx,0,:,:]
group1_v = ens_factor_uv[group1_idx,1,:,:]
group2_u = ens_factor_uv[group2_idx,0,:,:]
group2_v = ens_factor_uv[group2_idx,1,:,:]
group3_u = ens_factor_uv[group3_idx,0,:,:]
group3_v = ens_factor_uv[group3_idx,1,:,:]

# group1, group2, group3 각각의 모든 데이터를 가져와 평균 계산
group1_lon_list = [ssv_dict[key_time][member][datetime1]['lon'] for member in group1]
group1_lat_list = [ssv_dict[key_time][member][datetime1]['lat'] for member in group1]
group1_lon = np.mean([ssv_dict[key_time][member][datetime1]['lon'] for member in group1], axis=0)
group1_lat = np.mean([ssv_dict[key_time][member][datetime1]['lat'] for member in group1], axis=0)

group2_lon_list = [ssv_dict[key_time][member][datetime1]['lon'] for member in group2]
group2_lat_list = [ssv_dict[key_time][member][datetime1]['lat'] for member in group2]
group2_lon = np.mean([ssv_dict[key_time][member][datetime1]['lon'] for member in group2], axis=0)
group2_lat = np.mean([ssv_dict[key_time][member][datetime1]['lat'] for member in group2], axis=0)

group3_lon = np.mean([ssv_dict[key_time][member][datetime1]['lon'] for member in group3], axis=0)
group3_lat = np.mean([ssv_dict[key_time][member][datetime1]['lat'] for member in group3], axis=0)


# 공통 gridline 설정 함수
def set_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, linestyle='dotted', alpha=0.7)
    gl.xlocator = plt.FixedLocator(xticks)
    gl.ylocator = plt.FixedLocator(yticks)
    
    # 숫자 크기 키우기
    gl.xlabel_style = {'size': 20}  # 경도 숫자 크기
    gl.ylabel_style = {'size': 20}  # 위도 숫자 크기
    
    # 위쪽과 오른쪽 라벨 제거
    gl.top_labels = False
    gl.right_labels = False

figsize = (10 * latlon_ratio, 10)
# 위경도 5도 간격 설정
xticks = np.arange(130, 150, 5)  # 경도 130°E ~ 145°E, 5도 간격
yticks = np.arange(20, 40, 5)  # 위도 23°N ~ 38°N, 5도 간격

# 첫 번째 streamline 그림
fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': proj})
ax.set_extent([130, 145, 23, 38])
ax.coastlines()
set_gridlines(ax)  # gridline 설정 적용

ax.streamplot(lon_grid, lat_grid, group1_u.mean(axis=0), group1_v.mean(axis=0),
              transform=ccrs.PlateCarree(), density=1, color='black')

ax.scatter(group1_lon_list, group1_lat_list, color='blue', s=30, label="Group 1 Points")
plt.show()


# 두 번째 streamline 그림
fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': proj})
ax.set_extent([130, 145, 23, 38])
ax.coastlines()
set_gridlines(ax)  # gridline 설정 적용

ax.streamplot(lon_grid, lat_grid, group2_u.mean(axis=0), group2_v.mean(axis=0),
              transform=ccrs.PlateCarree(), density=1, color='black')

# ax.scatter(group2_lon, group2_lat, color='red', s=100, label="Group 2 Points")
ax.scatter(group2_lon_list, group2_lat_list, color='red', s=30, label="Group 2 Points")
plt.show()


# 세 번째 quiver 그림 (파란색 + scale box 추가)
fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': proj})
ax.set_extent([130, 145, 23, 38])
ax.coastlines()
set_gridlines(ax)  # gridline 설정 적용

# 벡터 차이
quiv = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4],
                 group2_u.mean(axis=0)[::4, ::4] - group1_u.mean(axis=0)[::4, ::4], 
                 group2_v.mean(axis=0)[::4, ::4] - group1_v.mean(axis=0)[::4, ::4], 
                 color='blue', transform=ccrs.PlateCarree(), scale = 100)

# scale 박스 추가
steer_key = ax.quiverkey(quiv, X=0.91, Y=0.97, U=5, label=f'5 m/s', labelpos='E', fontproperties={'size':13})
    
rect = patches.Rectangle((0.85, 0.94), 0.22, 0.07, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
ax.add_patch(rect)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': proj})
ax.set_extent([130, 145, 23, 38])
ax.set_extent([120, 155, 20, 45])
ax.coastlines()
set_gridlines(ax)  # gridline 설정 적용

ax.streamplot(lon_grid, lat_grid, group2_u.mean(axis=0) - group1_u.mean(axis=0), group2_v.mean(axis=0) - group1_v.mean(axis=0), density=1, linewidth=1, color='black', minlength=0.2, zorder = 0)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': proj})
ax.set_extent([130, 145, 23, 38])
ax.coastlines()
set_gridlines(ax)  # gridline 설정 적용

ax.scatter(mid_pos[:, 0], mid_pos[:, 1], color='green', s=15, zorder = 0)
ax.scatter(group1_lon_list, group1_lat_list, color='blue', s=30, label="Group 1 Points")
ax.scatter(group2_lon_list, group2_lat_list, color='red', s=30, label="Group 2 Points")
plt.show()

#%%
#stream plot으로 그려서 saddle point 찾아보기기
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.streamplot(lon_grid, lat_grid, group1_u.mean(axis=0), group1_v.mean(axis=0), density=20)
ax.scatter(group1_lon, group1_lat, color='green', s=100)
ax.coastlines('10m')
ax.set_extent([125,145, 20, 40])
gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 5))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.streamplot(lon_grid, lat_grid, group2_u.mean(axis=0), group2_v.mean(axis=0), density=20)
ax.scatter(group2_lon, group2_lat, color='green', s=100)
ax.coastlines('10m')
ax.set_extent([125,145, 20, 40])
gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 5))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.streamplot(lon_grid, lat_grid, group2_u.mean(axis=0)-group1_u.mean(axis=0), group2_v.mean(axis=0)-group1_v.mean(axis=0), density=20)
ax.scatter(group2_lon, group2_lat, color='green', s=100)
ax.coastlines('10m')
ax.set_extent([125,145, 20, 40])
gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 5))
plt.show()

#%%



fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
qui = ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], group2_u.mean(axis=0)[::4,::4]-group1_u.mean(axis=0)[::4,::4], group2_v.mean(axis=0)[::4,::4]-group1_v.mean(axis=0)[::4,::4], color = 'blue', scale = 100)
key = ax.quiverkey(qui, X=0.95, Y=0.97, U=5, label=f'{5}m/s', labelpos='E')
ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')
ax.scatter(mid_pos[:, 0], mid_pos[:, 1], color='green', s=10, zorder = 0)
ax.coastlines('10m')
ax.set_extent([125,160, 20, 38])

rect = patches.Rectangle((0.88, 0.94), 0.2, 0.15, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
ax.add_patch(rect)
gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 5))

plt.show()
#%%
# deformation flow가 평행이동 되었을 때 어떠한 차이를 보이는가가
import numpy as np
import matplotlib.pyplot as plt

# 회전된 벡터장 G(x,y) = (y, x)
def G(x, y):
    # 거리 r
    r = np.sqrt(x**2 + y**2)
    # 원점 근처가 상대적으로 세고, 멀어질수록 약해지도록 가우시안 스케일 사용
    # r=0에서 exp(-0) = 1로 최대값을 가지며, r가 커질수록 exp(-r^2/0.1)는 빠르게 0으로 수렴
    scale = np.log(1/r+3)
    
    # 원래 방향장 (-x, y)에 scale을 곱해서 
    # 원점 부근에서의 증가율을 낮추고 멀어질수록 더 약해지게 함
    u = -x + 5*y
    v = y + x
    # v = y + 0.2*x
    
    u, v = u/np.sqrt(u**2+v**2), v/np.sqrt(u**2+v**2)
    # u, v = u*scale, v*scale
    return u, v


# def G_shifted(x, y):
#     # 거리 r
#     # 원래 방향장 (-x, y)에 scale을 곱해서 
#     # 원점 부근에서의 증가율을 낮추고 멀어질수록 더 약해지게 함
#     u = -x + 10*y
#     v = y + x
#     # v = y + 0.2*x
    
#     u, v = u/np.sqrt(u**2+v**2), v/np.sqrt(u**2+v**2)
#     # u, v = u*scale, v*scale
#     return u, v

def G_rotated(x, y, theta_degrees=-20):
    U, V = G(x, y)
    theta = np.radians(theta_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    U_rot = U*cos_t - V*sin_t
    V_rot = U*sin_t + V*cos_t
    return U_rot, V_rot

delta_y = 0
delta_x = -3
x_vals = np.linspace(-5, 10, 30)
y_vals = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_vals, y_vals)

U, V = G(X, Y)              # 원래 회전된 벡터장
# U_shifted, V_shifted = G(X + delta_x, Y + delta_y)  # y축으로 평행이동한 벡터장
# U_shifted, V_shifted = G_shifted(X, Y)  # y축으로 평행이동한 벡터장
U_shifted, V_shifted = G_rotated(X + delta_x, Y + delta_y, theta_degrees=0)  # y축으로 평행이동한 벡터장

U_diff = U_shifted - U
V_diff = V_shifted - V

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Deformation Flow Difference delta_x: {delta_x}, delta_y: {delta_y}")

# 원래 벡터장
axs[0].quiver(X, Y, U, V, color='b')
axs[0].axhline(0, color='k', linewidth=1)
axs[0].axvline(0, color='k', linewidth=1)
axs[0].scatter(0,0,color='r',zorder=5)
# axs[0].set_title("Rotated Deformation: G(x,y)=(y,x)")
axs[0].set_aspect('equal', 'box')

# 평행이동한 벡터장
axs[1].quiver(X, Y, U_shifted, V_shifted, color='r')
axs[1].axhline(0, color='k', linewidth=1)
axs[1].axvline(0, color='k', linewidth=1)
axs[1].scatter(-delta_x,-delta_y,color='r',zorder=5)
# axs[1].set_title(f"G shifted by Δy={delta_y}")
axs[1].set_aspect('equal', 'box')

# 차이 벡터장
axs[2].quiver(X, Y, U_diff, V_diff, color='g', scale=10)
axs[2].axhline(0, color='k', linewidth=1)
axs[2].axvline(0, color='k', linewidth=1)
axs[2].scatter(0,0,color='r',zorder=5)
axs[2].scatter(3,0,color='b',zorder=5)
# axs[2].set_title("Difference: G_shifted - G")
axs[2].set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
