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
#! 태풍 경로 그리기
for key_time ,min_position in ssv_dict.items():
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    key_str = key_time.strftime("%m.%d.%HUTC")
    # ax.set_title(f'{key_time.strftime("%Y-%m-%d-%HUTC")} (+{predict_interval_list[-1]}h)', fontsize=20, loc = 'left')
    # ax.set_title(f'ENS{surface_str}{upper_str}{perturation_scale} Track\n{storm_name}', fontsize=20, loc = 'right')
    # ax.set_title(f'{storm_name}', fontsize=20, loc = 'right')
    ax.set_extent([120,155,20,45], crs=proj)
    setup_map(ax)

    ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'Best track', transform=ax.projection, zorder=10)
    model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=10)
    cbar = plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
    cbar.ax.tick_params(labelsize=15)

    
    for i in range(len(storm_time)):
        new_time = storm_time[i].strftime("%Y/%m/%d/%HUTC")
        if new_time.endswith('00UTC'):
            dx, dy = 5, -0.5  # 시간 나타낼 위치 조정
            new_lon, new_lat = storm_lon[i] + dx, storm_lat[i] + dy
            
            # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
            ax.text(storm_lon[i], new_lat, new_time[8:-6]
                    , horizontalalignment='right', verticalalignment='top', fontsize=15, zorder = 20, fontweight = 'bold')



    # for ens in range(ens_num):  
    for ens in ens_not_list[:10]:  


        lons = [pos['lon'] for _,pos in min_position[ens].items()]
        lats = [pos['lat'] for _,pos in min_position[ens].items()]
        min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
        pred_times = [pos for pos,_ in min_position[ens].items()]
        # print(ens)
        lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)

        #? 시간 표시 00UTC만 표시, 없앨듯

        # if ens == 0:
        #     lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)
        #     ax.scatter(lons, lats, c='red', linewidth=2, alpha=1, zorder=10, label = 'No perturbation')

        for i in range(len(pred_times)):
            if pred_times[i].hour == 0:
                ax.text(lons[i],lats[i], str(pred_times[i].day)
                    , horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder = 6)

        
    ax.legend(loc='upper right')


    lons_all = np.concatenate([np.array([pos['lon'] for _, pos in min_position[ens].items()]) for ens in ens_list])
    lats_all = np.concatenate([np.array([pos['lat'] for _, pos in min_position[ens].items()]) for ens in ens_list])


    xy = np.vstack([lons_all, lats_all])
    kde = gaussian_kde(xy)
    positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    f = np.reshape(kde(positions).T, lon_grid.shape)


    levels = np.linspace(0.0005, 0.015, 100)
    # cf = ax.contourf(lon_grid, lat_grid, f, levels=levels, transform=proj, cmap='jet')
    plt.show()
    # fig.savefig(f'{pangu_dir}/plot/Ensemble_track_{key_str}.png',bbox_inches='tight')
    
#%%
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)          #분석 시작 시점
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

fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
for predict_interval in np.arange(60,121,60):
# for predict_interval in [24,72]:
    datetime1 = key_time + timedelta(hours=int(predict_interval))

    mid_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
    ax.scatter(*zip(*mid_pos), c='green', s=10, alpha=0.5, label=f'{predict_interval}h', transform=ax.projection, zorder = 1)

for predict_interval in np.arange(120, 120+1,6):
    base_output_path = os.path.join(
        pangu_dir, 
        'output_data', 
        'steering_wind',
        key_str, 
        f'{perturation_scale}ENS{surface_str}{upper_str}', 
        f'{min(ens_list)}_{max(ens_list)}_{start_str}_{target_str}'
    )

    # Paths for saving the arrays
    u_mean_path = os.path.join(base_output_path, 'u_mean')
    v_mean_path = os.path.join(base_output_path, 'v_mean')
    u_mean = np.load(f'{u_mean_path}/{predict_interval}h{retro_opt}.npy')
    v_mean = np.load(f'{v_mean_path}/{predict_interval}h{retro_opt}.npy')


# ax.streamplot(lon_grid, lat_grid, u_mean, v_mean, color='gray', linewidth=2, density=0.7, arrowsize=2.5, transform=proj, zorder = 0)
# gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 20}
# gl.ylabel_style = {'size': 20}
# gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
# gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
# ax.coastlines('10m')

plt.show()

#%%
#! 가장 상관관계 높은 axis 구하기, 2nd main code
# 시간 지정
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)          #분석 시작 시점
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


for predict_interval in np.arange(0,120+1,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))
    ens_num_list = []

    for ens in ens_list:
        if (datetime1 in ssv_dict[key_time][ens]):
            ens_num_list.append(ens)

    print(predict_interval, len(ens_num_list))


# target_time 때도 살아있는 태풍만 추출
correlations = []
correlations_tar = []
correlations_opt = []
correlations_gg = []
correlations_df = []
correlations_all = []
ens_num_list = []

cov_var_convention = {}
dir_all = {}
sensitivity_sum = []
# for predict_interval in np.arange(start_time_range,total_time_range+1,6):
# for predict_interval in np.arange(start_time_range,120+1,6):
# for predict_interval in np.arange(54,total_time_range+1,6):
# for predict_interval in np.arange(start_time_range,36+1,6):
# for predict_interval in np.arange(start_time_range, 37, 6):
for predict_interval in np.arange(72,73,12):
# for predict_interval in [24,72]:
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
    mid_projection = pca_mid.transform(corr_pos_mid)[:, 0]  # 주축에 투영된 데이터 (1차원)
    mid_mean = np.mean(mid_projection)
    mid_distances = mid_projection - mid_mean  # 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산
    
    #distance 구하기
    corr_pos_tar = np.copy(tar_pos) #nearby를 고려하여 다시 target 부르기 PCA는 전체 데이터로 해야되므로 nearby 이전에 진행
    corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    projection = pca_tar.transform(corr_pos_tar)[:, 0]  # 주축에 투영된 데이터 (1차원)
    principal_component = pca_tar.components_[0]

    
    ensemble_mean = np.mean(projection)     # 투영된 데이터의 ensemble mean 계산
    distances = projection - ensemble_mean  # 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산
    # 각 앙상블 멤버의 거리를 저장
    ensemble_distances = {ens: distance for ens, distance in zip(ens_num_list, distances)}

    # 데이터를 추출합니다
    ens_pos = [(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]

    # 위도(lat) 기준으로 데이터를 정렬합니다
    ens_pos_sorted = sorted(ens_pos, key=lambda x: ensemble_distances[x[0]])

    # 위도가 가장 낮은 10개와 가장 높은 10개를 추출합니다
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

    
    uv_all_alt = {}
    uv_all_1 = {}
    uv_all_2 = {}
    for choosen_factor in choosen_factor_list:
        uv_all_alt[choosen_factor] = []
        uv_all_1[choosen_factor] = []
        uv_all_2[choosen_factor] = []
        for altitude in altitude_list:
            fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{key_str}/{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/'
            print(choosen_factor, altitude)     
            ens_factor_uv=[]
            total_remove_uv = []
            u_mean_each = []
            v_mean_each = []

            steer_quiver = {'u':[],'v':[]}
            steer_quiver_total = {'u':[],'v':[]}
            steer_quiver_g1 = {'u':[],'v':[]}
            steer_quiver_g2 = {'u':[],'v':[]}
            
            base_output_path = os.path.join(
                pangu_dir, 
                'output_data', 
                'steering_wind',
                key_str, 
                f'{perturation_scale}ENS{surface_str}{upper_str}', 
                f'{min(ens_list)}_{max(ens_list)}_{start_str}_{target_str}'
            )

            # Paths for saving the arrays
            ens_factor_uv_path = os.path.join(base_output_path, 'ens_factor_uv')
            total_remove_uv_path = os.path.join(base_output_path, 'total_remove_uv')
            u_mean_path = os.path.join(base_output_path, 'u_mean')
            v_mean_path = os.path.join(base_output_path, 'v_mean')
            
            if data_sign == 'n':
                for ens in ens_num_list:
                    center_lon, center_lat = ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']
                    
                    output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                    
                    if choosen_factor != 'steering_wind':
                        ens_factor_uv.append(met.met_data(choosen_factor, level = altitude))
                    else:
                        u_list = []
                        v_list = []
                        
                        
                        if len(steer_pres) > 1:
                            dis_cal_sign = 'y'
                            final_mask = None
                            for steer_altitude in steer_pres:
                                
                                div = met.divergence(level = steer_altitude)
                                vort = met.vorticity(level = steer_altitude)
                                vort_850 = met.vorticity(level = 850)
                                
                                if steering_sign == 'y':
                                    ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850, dis_cal_sign = dis_cal_sign, final_mask = final_mask)
                                    dis_cal_sign = 'n'
                                    u_ty, v_ty, final_mask = ty_wind.solve()
                                    u_list.append(met.met_data('u', level = steer_altitude)-u_ty)
                                    v_list.append(met.met_data('v', level = steer_altitude)-v_ty)
                                    # u_list.append(met.met_data('u', level = steer_altitude))
                                    # v_list.append(met.met_data('v', level = steer_altitude))

                                else:
                                    u_list.append(met.met_data('u', level = steer_altitude))
                                    v_list.append(met.met_data('v', level = steer_altitude))
                                    
                            u,v = np.zeros(np.shape(u_list[0])), np.zeros(np.shape(u_list[0]))
                            
                        
                            for i in range(len(steer_pres)-1):
                                u += (u_list[i]+u_list[i+1])/2*(steer_pres[i]-steer_pres[i+1])
                                v += (v_list[i]+v_list[i+1])/2*(steer_pres[i]-steer_pres[i+1])
                            
                            u/=np.ptp(steer_pres)
                            v/=np.ptp(steer_pres)
                            
                        else:
                            div = met.divergence(level = steer_pres[0])
                            vort = met.vorticity(level = steer_pres[0])
                            
                            if steering_sign == 'y':
                                ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850)
                                u_ty, v_ty = ty_wind.solve()
                                u = met.met_data('u', level = steer_pres[0]-u_ty)
                                v = met.met_data('v', level = steer_pres[0]-v_ty)
                                
                            else:
                                u = met.met_data('u', level = steer_pres[0])
                                v = met.met_data('v', level = steer_pres[0])
                        
                        u_mean_each.append(u)
                        v_mean_each.append(v)
                        # ens_factor.append(u * best_direction[0] + v * best_direction[1])
                        ens_factor_uv.append([u,v])
                        total_remove_uv.append([u_list, v_list])
                        
                        if ens == 0:
                            u0 = u
                            v0 = v
                            
                u_mean = np.mean(np.array(u_mean_each), axis=0)
                v_mean = np.mean(np.array(v_mean_each), axis=0)
            
                # Ensure the figure directory exists
                # fig_path = os.path.join(fig_dir, f'{altitude}hPa')
                # os.makedirs(fig_path, exist_ok=True)

                # Base path for output data


            
            else:
                ens_factor_uv = np.load(f'{ens_factor_uv_path}/{predict_interval}h{retro_opt}.npy')
                total_remove_uv = np.load(f'{total_remove_uv_path}/{predict_interval}h{retro_opt}.npy')
                
                u_mean = np.load(f'{u_mean_path}/{predict_interval}h{retro_opt}.npy')
                v_mean = np.load(f'{v_mean_path}/{predict_interval}h{retro_opt}.npy')
                
                if steer_uni_alt in steer_pres:
                    ens_factor_uv = total_remove_uv[:,:,steer_pres.index(steer_uni_alt),:,:]
                    u_mean = np.mean(ens_factor_uv[:,0,:,:], axis=0)
                    v_mean = np.mean(ens_factor_uv[:,1,:,:], axis=0)
                
                if choosen_factor == 'u':
                    ens_factor_uv = ens_factor_uv[:,0,:,:]
                    total_remove_uv = total_remove_uv[:,0,:,:,:]
                    
                elif choosen_factor == 'v':
                    ens_factor_uv = ens_factor_uv[:,1,:,:]
                    total_remove_uv = total_remove_uv[:,1,:,:,:]
            
                if nearby_sign == 'y':
                    if data_sign == 'y':
                        ens_factor_uv = np.array(ens_factor_uv)[group_idx]
                    
                    if steering_sign == 'y':
                        u_mean = np.mean(ens_factor_uv[:,0,:,:], axis=0)
                        v_mean = np.mean(ens_factor_uv[:,1,:,:], axis=0)
            
            
            non_uv_mean = np.mean(ens_factor_uv, axis=0)
            
                     
            
            #! 가장 상관관계 높은 axis 구하기
            angles = np.linspace(0, 2 * np.pi, 1441)
            directions = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
            
            # 상관관계 계산
            best_correlation = -1
            best_direction = None
            dir_corr = []
            for direction in directions:
                # 중간 위치 데이터를 방향 벡터에 사영
                mid_projection = mid_pos @ direction
                # 목표 위치 데이터와의 상관관계 계산
                correlation = np.corrcoef(mid_projection, tar_proj[:, 0])[0, 1]  # 상관관계는 X축만 고려
                dir_corr.append(correlation)
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_direction = direction

            # 상관관계가 최대인 방향 찾기
            max_index = np.argmax(dir_corr)
            best_correlation = dir_corr[max_index]
            correlation_opt = best_correlation
            best_direction = directions[max_index]

            # 결과 출력
            print(f"최대 상관관계: {best_correlation}, 방향: {best_direction}")
            

            rad_pca = np.arctan2(pca_mid.components_[0, 1], pca_mid.components_[0, 0])
            rad_tar = np.arctan2(pca_tar.components_[0, 1], pca_tar.components_[0, 0])
            
            # mid_df = mid_pos @ df_direction
            # correlation_df = np.corrcoef(mid_df, tar_proj[:, 0])[0, 1]
            # correlations_df.append((predict_interval, correlation_df))
            
            correlations_opt.append((predict_interval, best_correlation))
            
            correlation = np.corrcoef(mid_proj[:, 0], tar_proj[:, 0])[0, 1]
            correlations.append((predict_interval, correlation))
            
            
            correlation_tar = np.corrcoef(mid2tar[:, 0], tar_proj[:, 0])[0, 1]
            correlations_tar.append((predict_interval, correlation_tar))
            
            
            #! 상관관계 그래프 그리기
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)  # 극 좌표계 사용

            # 원형 그래프에 데이터 플로팅
            ax.plot(angles, dir_corr, label='Correlation', linewidth=2)  # 상관관계 선 그리기
            # 최대 상관관계 위치에 화살표 표시
            max_corr_radian = angles[max_index]
            
            # ax.quiver(max_corr_radian, 0, 0, best_correlation, angles='xy', scale_units='xy', scale=1, color='red', label='Max Correlation')
            ax.quiver(max_corr_radian, -1, 0, best_correlation + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='Max Corr')
            # PCA 방향에 화살표 표시
            pca_value = dir_corr[int(rad_pca /np.pi /2 * len(angles))]
            ax.quiver(rad_pca, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='green', label='PCA Direction')
            
            
            # pca_value = dir_corr[int(rad_df /np.pi /2 * len(angles))]
            # ax.quiver(rad_df, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='PCA Direction')
            ax.axvline(rad_tar, color='red', linewidth=2)
            
            ax.set_ylim(-1, 1)
            ax.set_theta_zero_location('E')  # 0도를 북쪽(위)으로 설정
            ax.set_yticks(np.arange(-1, 1.1, 0.5))

            # 저장 및 보여주기
            if not os.path.exists(f'{fig_dir}/Projection_direction'):
                os.makedirs(f'{fig_dir}/Projection_direction')
            else:
                pass
            plt.savefig(f'{fig_dir}/Projection_direction/{predict_interval}h.png')
            plt.close()


            # z_500 데이터를 numpy 배열로 변환
            ens_factor = np.array(ens_factor_uv)
            
            if steering_sign == 'y':
                if axis_opt == 'opt':
                    ens_factor = ens_factor[:,0,:,:]*best_direction[0] + ens_factor[:,1,:,:]*best_direction[1]
                elif axis_opt == 'tar':
                    ens_factor = ens_factor[:,0,:,:]*pca_tar.components_[0, 0] + ens_factor[:,1,:,:]*pca_tar.components_[0, 1]
                elif axis_opt == 'mid':
                    ens_factor = ens_factor[:,0,:,:]*pca_mid.components_[0, 0] + ens_factor[:,1,:,:]*pca_mid.components_[0, 1]
                elif axis_opt == 'lon':
                    ens_factor = ens_factor[:,0,:,:]
                elif axis_opt == 'lat':
                    ens_factor = ens_factor[:,1,:,:]
                elif axis_opt == 'quiver':
                    ens_factor_quiver = (ens_factor[:,0,:,:], ens_factor[:,1,:,:])
                
                
                
                if axis_opt == 'quiver':
                    cov_var_ratio = {}
                    cov_var_quiver = {'u':[],'v':[]}
                    corr_mid = {'u':[],'v':[]}
                    for ens_factor, uv_key in zip(ens_factor_quiver,['u','v']):
                        ens_factor_std = np.std(ens_factor, axis=0)
                        ens_factor = (ens_factor - np.mean(ens_factor, axis=0)) / ens_factor_std

                        # 공분산 및 분산 계산
                        cov_matrix = np.zeros_like(ens_factor[0])
                        var_matrix = np.zeros_like(ens_factor[0])


                        for i in range(ens_factor.shape[1]):  # lat 방향
                            for j in range(ens_factor.shape[2]):  # lon 방향
                                cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
                                var_matrix[i, j] = np.var(ens_factor[:,i,j])


                    

                        # NaN 또는 Inf 값을 0으로 대체
                        cov_var_ratio[uv_key] = np.nan_to_num(cov_matrix / var_matrix * 111)
                        
                        #! mid_proj를 사용하여 중간 위치와 steering wind간의 상관관계 구하기기
                        # ── (1)  변수 준비 ───────────────────────────────────────────
                        x = mid_proj[:, 0]                 # (2529,)   ← 1차원 벡터
                        y = ens_factor.reshape(x.size, -1) # (2529, 161*241)

                        # ── (2)  평균·표준편차 ───────────────────────────────────────
                        x_mean  = x.mean()
                        x_std   = x.std()

                        y_mean  = y.mean(axis=0)           # (npoint,)
                        y_std   = y.std(axis=0)

                        # ── (3)  공분산 → 피어슨 r ──────────────────────────────────
                        cov  = ((y - y_mean) * (x[:, None] - x_mean)).sum(axis=0) / (len(x) - 1)
                        corr = cov / (x_std * y_std)       # (npoint,)

                        # ── (4)  다시 (lat, lon) 그리드로 ───────────────────────────
                        corr_grid = corr.reshape(161, 241)
                        corr_grid = np.nan_to_num(corr_grid)  # std=0인 곳 처리
                        corr_mid[uv_key] = corr_grid
                        

                    for k, mp in enumerate(mid_pos):
                        dis = haversine_distance(lat_grid, lon_grid, np.ones_like(lat_grid)*mp[1], np.ones_like(lat_grid)*mp[0])
                        cov_var_quiver['u'].append(np.mean(cov_var_ratio['u'][dis <= 333]))
                        cov_var_quiver['v'].append(np.mean(cov_var_ratio['v'][dis <= 333]))
                        steer_quiver['u'].append(np.mean(ens_factor_uv[k,0,:,:][dis <= 333]))
                        steer_quiver['v'].append(np.mean(ens_factor_uv[k,1,:,:][dis <= 333]))
                        steer_quiver_total['u'].append(ens_factor_uv[k,0,:,:])
                        steer_quiver_total['v'].append(ens_factor_uv[k,1,:,:])

                    for g1_idx, g2_idx in zip(group1_idx, group2_idx):
                        steer_quiver_g1['u'].append(ens_factor_uv[g1_idx,0,:,:])
                        steer_quiver_g1['v'].append(ens_factor_uv[g1_idx,1,:,:])
                        steer_quiver_g2['u'].append(ens_factor_uv[g2_idx,0,:,:])
                        steer_quiver_g2['v'].append(ens_factor_uv[g2_idx,1,:,:])

            else:  
                ens_factor_std = np.std(ens_factor, axis=0)
                ens_factor_quiver = (ens_factor - np.mean(ens_factor, axis=0)) / ens_factor_std

                # 공분산 및 분산 계산
                cov_matrix = np.zeros_like(ens_factor[0])
                var_matrix = np.zeros_like(ens_factor[0])

                for i in range(ens_factor.shape[1]):  # lat 방향
                    for j in range(ens_factor.shape[2]):  # lon 방향
                        cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
                        var_matrix[i, j] = np.var(ens_factor[:,i,j])

            
                # 공분산/분산 비율 계산
                cov_var_ratio = cov_matrix / var_matrix * 111 #! 이거 한번 확인할 필요 있음!


                # NaN 또는 Inf 값을 0으로 대체
                cov_var_ratio = np.nan_to_num(cov_var_ratio)
        
        cov_var_convention[predict_interval] = {
            'map': cov_var_ratio,
            'inner': cov_var_quiver,
        }

#%%
            #! mid_proj를 사용하여 중간 위치와 steering wind간의 상관관계 구하기기
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([130,150,20,40], crs=proj)
            # contour = ax.contourf(lon_grid, lat_grid, corr_mid['v'], levels=np.linspace(0, 1, 21), transform=ccrs.PlateCarree())
            cq = ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], corr_mid['u'][::4,::4], corr_mid['v'][::4,::4], transform=ccrs.PlateCarree())
            ax.quiverkey(cq,
             X=-.065, Y=0.97,      # 위치 (axes 좌표)
             U=1,                # 참조 벡터 크기(데이터 단위)   ← 원하는 값으로
             label='$|r|=1$', # 라벨 텍스트
             labelpos='S',         # 라벨을 화살표 오른쪽에
             coordinates='axes',
             fontproperties={'size': 12, 'weight': 'bold'})
            # plt.colorbar(contour, ax=ax, shrink = 1)
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=1, s=2.5, c='green', zorder = 11)
            # setup_map(ax, back_color='n')
            ax.coastlines(resolution='10m', color='black', linewidth=1)
            gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12}
            gl.ylabel_style = {'size': 12}
        
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            ax.streamplot(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], density = 1, linewidth=0.5, color = 'gray')
#%%
            
            # 지도에 결과 표시
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([125,160,20,40], crs=proj)
            if steering_sign != 'y':
                contour = ax.contourf(lon_grid, lat_grid, cov_var_ratio, cmap=pwp, levels=np.linspace(-200, 200, 17), transform=ccrs.PlateCarree())
                cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}) / Var({choosen_factor})', shrink = 1)
                cbar.locator = mticker.MultipleLocator(50)  # Set the colorbar ticks to have an interval of 0.5
                cbar.update_ticks()
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=1, s=2.5, c='green', zorder = 11)
            # setup_map(ax, back_color='n')
            ax.coastlines(resolution='10m', color='black', linewidth=1)
            gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12}
            gl.ylabel_style = {'size': 12}
        
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            
           
            if steering_sign == 'y':
                # if axis_opt != 'quiver':
                # ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5, color = 'black')
                ax.streamplot(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], density = 1, linewidth=0.5, color = 'gray')

                        
                if axis_opt == 'opt':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], best_direction[0], best_direction[1], scale=20, color='b', width=0.003, label='Principal Axis')
                elif axis_opt == 'tar':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1], scale=20, color='r', width=0.003, label='Principal Axis')
                elif axis_opt == 'mid':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_mid.components_[0, 0], pca_mid.components_[0, 1], scale=20, color='g', width=0.003, label='Principal Axis')
                elif axis_opt == 'lon':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 1, 0, scale=20, color='black', width=0.003, label='Principal Axis')
                elif axis_opt == 'lat':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 0, 1, scale=20, color='black', width=0.003, label='Principal Axis')
                elif axis_opt == 'quiver':
                    # 태풍 영역 안에 들어오는 sensitivity의 방향 평균을 구하기, 정규화
                    u_mean_inner = np.mean(cov_var_quiver['u'])
                    v_mean_inner = np.mean(cov_var_quiver['v'])
                    u_steer = np.mean(steer_quiver['u'])
                    v_steer = np.mean(steer_quiver['v'])


                    qui_length = 500
                    
                    mean_qui = ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], u_mean_inner, v_mean_inner, scale=qui_length*4, color='gray', width=0.003, zorder = 12)
                    qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], cov_var_ratio['u'][::4, ::4], cov_var_ratio['v'][::4, ::4], scale=qui_length*20, color='blue', width=0.003)
                    # 먼저, quiverkey를 추가합니다.
                    key = ax.quiverkey(qui, X=0.9, Y=0.97, U=qui_length, label=f'{qui_length}km', labelpos='E')
                    mean_key = ax.quiverkey(mean_qui, X=0.9, Y=0.93, U=qui_length/5, label=f'{int(qui_length/5)}km', labelpos='E')

                    steer_qui = ax.quiver(pca_mid.mean_[0],  pca_mid.mean_[1], u_steer, v_steer, scale=100, color='orange', width=0.003, zorder = 12)
                    steer_key = ax.quiverkey(steer_qui, X=0.9, Y=0.89, U=5, label=f'5 m/s', labelpos='E')
                    
                    rect = patches.Rectangle((0.84, 0.87), 0.16, 0.15, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
                    ax.add_patch(rect)
                    
                    cov_u_dir, cov_v_dir = u_mean_inner/np.sqrt(u_mean_inner**2+v_mean_inner**2), v_mean_inner/np.sqrt(u_mean_inner**2+v_mean_inner**2)
                    cov_proj = mid_pos @ np.array([cov_u_dir, cov_v_dir])
                    
                    correlation = np.corrcoef(cov_proj, tar_proj[:, 0])[0, 1]
                    correlations_all.append((predict_interval, correlation))
                    dir_all[predict_interval]={'u': u_mean_inner, 'v': v_mean_inner}
                    
                    
                    
            else:
                if choosen_factor == 'z':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,15001,60), colors='black')
                elif choosen_factor == 't':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(200,401,5), colors='black')
                elif choosen_factor == 'q':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,21,1),    colors='black')
                elif choosen_factor == 'u':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(-100,101,10), colors='black')
                elif choosen_factor == 'v':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(-100,101,10), colors='black')
                
                cax.clabel()
            
                
                
            ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')

            # fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{start_str}_{target_str}_z/{choosen_factor}/'
            if not os.path.exists(f'{fig_dir}/{choosen_factor}/{altitude}hPa'):
                os.makedirs(f'{fig_dir}/{choosen_factor}/{altitude}hPa')
            else:
                pass
            if steering_sign == 'y':
                
                if steer_uni_alt in steer_pres:
                    if not os.path.exists(f'{fig_dir}/{choosen_factor}/{steer_uni_alt}hPa'):
                        os.makedirs(f'{fig_dir}/{choosen_factor}/{steer_uni_alt}hPa')
                    else:
                        pass
                    fig.savefig(f'{fig_dir}/{choosen_factor}/{steer_uni_alt}hPa/{predict_interval}h.png', bbox_inches='tight')    
                else:
                    fig.savefig(f'{fig_dir}/{choosen_factor}/{predict_interval}h.png',bbox_inches='tight')
            else: 
                fig.savefig(f'{fig_dir}/{choosen_factor}/{altitude}hPa/{predict_interval}h.png', bbox_inches='tight')
            plt.close()
            #%%

            if not os.path.exists(f'{fig_dir}/{choosen_factor}/wind_field/'):
                os.makedirs(f'{fig_dir}/{choosen_factor}/wind_field/')
            else:
                pass
            # 지도에 결과 표시
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([130,145,23,38], crs=proj)
            ax.coastlines(resolution='10m', color='black', linewidth=1)
            # ax.set_extent([pca_mid.mean_[0]-5,pca_mid.mean_[0]+5,pca_mid.mean_[1]-5,pca_mid.mean_[1]+5], crs=proj)
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=1, s=15, c='green', zorder = 11, label = 'TCs location')
            ax.scatter(mid_pos[group1_idx, 0], mid_pos[group1_idx, 1], s=30, c='blue', zorder = 11, label = 'SW members')
            ax.scatter(mid_pos[group2_idx, 0], mid_pos[group2_idx, 1], s=30, c='red', zorder = 11, label = 'NE members')
            # ax.quiver(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis = 0), np.mean(steer_quiver_total['v'], axis = 0), scale = 100)
            ax.streamplot(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis=0), np.mean(steer_quiver_total['v'], axis=0), density=1, linewidth=1, color='black', minlength=0.3)
            # 위경도 숫자만 표시하기
            # gridlines 추가, 라벨은 표시하되 그리드 라인은 그리지 않음

            
            gl = ax.gridlines(draw_labels=True)

            # 축에 정수로 라벨만 표시하고 그리드 라인은 비활성화
            gl.xlines = False
            gl.ylines = False

            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 24}
            gl.ylabel_style = {'size': 24}
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            ax.tick_params(axis='both', labelsize=24)
            # ax.tick_params(labelsize=24)  # 숫자 크기를 14pt로 설정 (원하는 크기로 조절 가능)
            # ax.legend(loc = 'upper right', fontsize = 15)
            fig.savefig(f'{fig_dir}/{choosen_factor}/wind_field/{predict_interval}h_both.png',bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([130,145,23,38], crs=proj)
            ax.coastlines(resolution='10m', color='black', linewidth=1)
            # ax.set_extent([pca_mid.mean_[0]-5,pca_mid.mean_[0]+5,pca_mid.mean_[1]-5,pca_mid.mean_[1]+5], crs=proj)
            ax.scatter(mid_pos[group1_idx, 0], mid_pos[group1_idx, 1], alpha=1, s=30, c='blue', zorder = 11)
            # ax.quiver(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis = 0), np.mean(steer_quiver_total['v'], axis = 0), scale = 100)
            ax.streamplot(lon_grid, lat_grid, np.mean(steer_quiver_g1['u'], axis=0), np.mean(steer_quiver_g1['v'], axis=0), density=1, linewidth=1, color='black', minlength=0.2)
            # ax.gridline()
            # 위경도 숫자만 표시하기
            # gridlines 추가, 라벨은 표시하되 그리드 라인은 그리지 않음
            gl = ax.gridlines(draw_labels=True)

            # 축에 정수로 라벨만 표시하고 그리드 라인은 비활성화
            gl.xlines = False
            gl.ylines = False

            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 24}
            gl.ylabel_style = {'size': 24}
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            ax.tick_params(labelsize=14)  # 숫자 크기를 14pt로 설정 (원하는 크기로 조절 가능)
            fig.savefig(f'{fig_dir}/{choosen_factor}/wind_field/{predict_interval}h_g1.png',bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([130,145,23,38], crs=proj)
            ax.coastlines(resolution='10m', color='black', linewidth=1)
            # ax.set_extent([pca_mid.mean_[0]-5,pca_mid.mean_[0]+5,pca_mid.mean_[1]-5,pca_mid.mean_[1]+5], crs=proj)
            ax.scatter(mid_pos[group2_idx, 0], mid_pos[group2_idx, 1], alpha=1, s=30, c='red', zorder = 11)
            # ax.quiver(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis = 0), np.mean(steer_quiver_total['v'], axis = 0), scale = 100)
            ax.streamplot(lon_grid, lat_grid, np.mean(steer_quiver_g2['u'], axis=0), np.mean(steer_quiver_g2['v'], axis=0), density=1, linewidth=1, color='black', minlength=0.2)
            # ax.gridline()
            # 위경도 숫자만 표시하기
            # gridlines 추가, 라벨은 표시하되 그리드 라인은 그리지 않음
            gl = ax.gridlines(draw_labels=True)

            # 축에 정수로 라벨만 표시하고 그리드 라인은 비활성화
            gl.xlines = False
            gl.ylines = False

            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 24}
            gl.ylabel_style = {'size': 24}
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            ax.tick_params(labelsize=14)  # 숫자 크기를 14pt로 설정 (원하는 크기로 조절 가능)
            fig.savefig(f'{fig_dir}/{choosen_factor}/wind_field/{predict_interval}h_g2.png',bbox_inches='tight')
            plt.close()
            
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([120,160,10,40], crs=proj)
            # ax.set_extent([pca_mid.mean_[0]-5,pca_mid.mean_[0]+5,pca_mid.mean_[1]-5,pca_mid.mean_[1]+5], crs=proj)
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=1, s=1, c='green', zorder = 11)
            # ax.quiver(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis = 0), np.mean(steer_quiver_total['v'], axis = 0), scale = 100)
            ax.streamplot(lon_grid, lat_grid, np.mean(steer_quiver_g2['u'], axis=0)-np.mean(steer_quiver_g1['u'], axis=0), np.mean(steer_quiver_g2['v'], axis=0) - np.mean(steer_quiver_g1['v'], axis=0), density=1, linewidth=1, color='black', minlength=0.2)
            ax.coastlines()
            # ax.gridline()
            # 위경도 숫자만 표시하기
            # gridlines 추가, 라벨은 표시하되 그리드 라인은 그리지 않음
            gl = ax.gridlines(draw_labels=True)

            # 축에 정수로 라벨만 표시하고 그리드 라인은 비활성화
            gl.xlines = False
            gl.ylines = False

            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 24}
            gl.ylabel_style = {'size': 24}
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            ax.tick_params(labelsize=14)  # 숫자 크기를 14pt로 설정 (원하는 크기로 조절 가능)
            fig.savefig(f'{fig_dir}/{choosen_factor}/wind_field/{predict_interval}h_large.png',bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
            ax.set_extent([120,140,20,40], crs=proj)
            ax.coastlines()
            # ax.quiver(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis = 0), np.mean(steer_quiver_total['v'], axis = 0), scale = 100)
            ax.streamplot(lon_grid, lat_grid, np.mean(steer_quiver_total['u'], axis=0), np.mean(steer_quiver_total['v'], axis=0), density=1, linewidth=1, color='black', minlength=0.3)
            # 위경도 숫자만 표시하기
            # gridlines 추가, 라벨은 표시하되 그리드 라인은 그리지 않음

            
            gl = ax.gridlines(draw_labels=True)

            # 축에 정수로 라벨만 표시하고 그리드 라인은 비활성화
            gl.xlines = False
            gl.ylines = False

            gl.top_labels = False
            gl.right_labels = False
            ax.tick_params(axis='both', labelsize=24)
            # ax.tick_params(labelsize=24)  # 숫자 크기를 14pt로 설정 (원하는 크기로 조절 가능)
            # ax.legend(loc = 'upper right', fontsize = 15)
            fig.savefig(f'{fig_dir}/{choosen_factor}/wind_field/{predict_interval}h_total_mean.png',bbox_inches='tight')
            plt.close()

        
#%%

# 상관관계 시각화
predict_intervals, corr_values = zip(*correlations)
plt.plot(predict_intervals, np.abs(corr_values), marker='o', label = 'Each spread')
# predict_intervals, corr_values = zip(*correlations_tar)
# plt.plot(predict_intervals, np.abs(corr_values), marker='^', label = 'Projected final axis')
predict_intervals, corr_values = zip(*correlations_opt)
plt.plot(predict_intervals, np.abs(corr_values), marker='x', label = 'Optimized')
predict_intervals, corr_values = zip(*correlations_all)
plt.plot(predict_intervals, np.abs(corr_values), marker='^', label = 'All position')
predict_intervals, corr_values = zip(*correlations_cov)
plt.plot(predict_intervals, np.abs(corr_values), marker='*', label = 'Same position')
# predict_intervals, corr_values = zip(*correlations_df)
# plt.plot(predict_intervals, corr_values, marker='^', label = 'Projection_deformation')
plt.ylim(0, 1)  
plt.xlim(start_time_range, total_time_range)  
# plt.xlim(24, 120)  
plt.grid(True)

# x축 gridline 설정
plt.xticks(np.arange(start_time_range, total_time_range+1, 24))
plt.legend(loc = 'lower right')
plt.show()
fig.savefig(f'{fig_dir}/Position_correlation_{start_str}_{target_str}.png',bbox_inches='tight')

#%%
print(dir_all)
#%%
#! only nearby, quiver, 위치 변동 아예 없애기
from numba import njit, prange
@njit(parallel=True)
def compute_cov_var_ratio(distances, ens_data):
    n_lat, n_lon = ens_data.shape[1], ens_data.shape[2]
    n_ens = ens_data.shape[0]
    cov_matrix = np.zeros((n_lat, n_lon))
    var_matrix = np.zeros((n_lat, n_lon))

    for i in prange(n_lat):
        for j in prange(n_lon):
            x = distances
            y = ens_data[:, i, j]
            x_mean = x.mean()
            y_mean = y.mean()
            cov = np.sum((x - x_mean) * (y - y_mean)) / (n_ens - 1)
            var = np.sum((y - y_mean) ** 2) / (n_ens - 1)
            cov_matrix[i, j] = cov
            var_matrix[i, j] = var

    # for i in range(ens_factor.shape[1]):  # lat 방향
    #     for j in range(ens_factor.shape[2]):  # lon 방향
    #         cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
    #         var_matrix[i, j] = np.var(ens_factor[:,i,j])

    return cov_matrix / (var_matrix + 1e-6) * 111

key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)          #분석 시작 시점
target_time = datetime(2022,9,1,0)          #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")

total_time_range = int((target_time - key_time).total_seconds() / 3600)
start_time_range = int((start_time  - key_time).total_seconds() / 3600)

# 변수 지정
nearby_sign = 'y'                           #가까운 태풍만 추출
steering_sign = 'y'                         #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
distance_threshold = 0                      #가까운 태풍의 거리
choosen_factor_list = ['t','q']             #분석할 변수
altitude_list = [250,500,850]               #분석할 고도
# altitude_list = [1000,700,300,200]
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x
axis_opt = 'quiver'                         #axis 뭘로 잡을지, opt: 위치 상관관계 최대인 axis, tar: 최종 위치의 axis, mid: 중간 위치의 axis, lon: 경도, lat: 위도
data_sign = 'y'                             #기존의 데이터를 사용할 것인지, n이면 새로 구함
min_mem_threshold = 20                       #최소 멤버 수

if nearby_sign == 'y':
    nearby_sign_name = '_nearby'
else:
    nearby_sign_name = ''
    

# steering_sign이 y면 steering wind에 대해서만 구하기
if steering_sign == 'y':
    altitude_list = ['850_200']
    choosen_factor_list = ['steering_wind']



# target_time 때도 살아있는 태풍만 추출
correlations = []
correlations_same = []


# dir_near = {}
group_mem_dict = {}
cov_var_total = {}
cov_var_nearby = {}
# for predict_interval in np.arange(start_time_range,total_time_range+1,6):
# for predict_interval in np.arange(78,total_time_range+1,6):
# for predict_interval in np.arange(start_time_range,72+1,6):
# for predict_interval in np.arange(start_time_range,72+1,6):
for predict_interval in np.arange(24,25,6):
# for predict_interval in np.arange(18,19,6):
# for predict_interval in [24, 72]:
# for predict_interval in np.arange(120,124,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))

    for choosen_factor in choosen_factor_list:
        for altitude in altitude_list:
            print(altitude)
            cov_var_inner = {'u':[],'v':[]}
            cov_var_inner_each = {'u':[],'v':[]}
            cov_var_map = {'u':[],'v':[]}
            
            steer_quiver = {'u':[],'v':[]}
            steer_quiver_each = {'u':[],'v':[]}
            
            cov_var_nonuv = []
            u_mean = []
            v_mean = []
            non_uv_mean = []
            
            
            group_num=1
            group_mem=0
            group_mem_dict[predict_interval] = []
            mid_pos_group = []
            
            
            while True:
                ens_num_list = []
                
                for ens in ens_list:
                    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
                        ens_num_list.append(ens)
                
                
                # 데이터를 추출합니다
                ens_pos = [(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]

                # 위도(lat) 기준으로 데이터를 정렬합니다
                pos_sorted_by_lat = sorted(ens_pos, key=lambda x: x[2])  # x[2]는 위도를 나타냅니다

                # 위도가 가장 낮은 10개와 가장 높은 10개를 추출합니다
                group1 = pos_sorted_by_lat[:50]  # 가장 낮은 10개
                group2 = pos_sorted_by_lat[-50:]  # 가장 높은 10개
                group3 = pos_sorted_by_lat[len(pos_sorted_by_lat) // 2 - 5:len(pos_sorted_by_lat) // 2 + 5]
                
                # group1과 group2에 있는 ens 번호만 추출합니다
                group1 = [item[0] for item in group1]
                group2 = [item[0] for item in group2]
                group3 = [item[0] for item in group3]
                
                mid_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
                tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]
                gg_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in group1 + group2]
                gg_tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in group1 + group2]
                
                # NumPy 배열로 변환
                mid_pos, tar_pos, gg_pos, gg_tar_pos = np.array(mid_pos), np.array(tar_pos), np.array(gg_pos), np.array(gg_tar_pos)
                total_mean_lon = np.mean(mid_pos[:,0])
                total_mean_lat = np.mean(mid_pos[:,1])
                
                
                #타겟 위치의 PCA
                corr_pos_tar = np.copy(tar_pos)
                corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
                pca_tar = PCA(n_components=1)
                pca_tar.fit(corr_pos_tar)
                pca_tar.mean_[0] = pca_tar.mean_[0] / np.cos(np.radians(pca_tar.mean_[1])) + np.mean(tar_pos[:, 0])

                # 중간 위치의 PCA
                corr_pos_mid = np.copy(mid_pos)
                corr_pos_mid[:, 0] = (mid_pos[:, 0]-np.mean(mid_pos[:, 0])) * np.cos(np.radians(mid_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
                pca_mid = PCA(n_components=1)
                pca_mid.fit(corr_pos_mid)
                pca_mid.mean_[0] = pca_mid.mean_[0] / np.cos(np.radians(pca_mid.mean_[1])) + np.mean(mid_pos[:, 0])


                mid_total_proj = pca_mid.transform(mid_pos)
                tar_total_proj = pca_tar.transform(tar_pos)
                mid_pos_total = np.copy(mid_pos)
                
            
                #마지막 숫자는 최소 그룹 멤버 개수, 5명 이하면 그룹으로 인정하지 않음 그리고 while문 탈출
                result = find_large_groups(mid_pos, ens_num_list, tar_pos, nearby_sign, distance_threshold, group_num, min_mem_threshold)
                group_num+=1
                if not result:
                    print(f"No more groups with at least {min_mem_threshold} members found.")
                    break
                
                tar_pos, mid_pos, group_idx = result
                group_mem+=len(group_idx)
                group_mem_dict[predict_interval].append(group_idx)
                mid_pos_group.append(mid_pos[0])
                # if np.all(mid_pos == mid_pos[0]):
                #     print("모든 내용물이 동일합니다.")
                # else:
                #     print("모든 내용물이 동일하지 않습니다.")
                print(f'{predict_interval}h', f'총 개수:{len(ens_num_list)}, 누적 개수:{group_mem}, 현재 그룹 개수:{len(group_idx)}')
                    
                
                
                #distance 구하기
                corr_pos_tar = np.copy(tar_pos) #nearby를 고려하여 다시 target 부르기 PCA는 전체 데이터로 해야되므로 nearby 이전에 진행
                corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
                projection = pca_tar.transform(corr_pos_tar)[:, 0]  # 주축에 투영된 데이터 (1차원)
                principal_component = pca_tar.components_[0]
                # 투영된 데이터의 ensemble mean 계산
                ensemble_mean = np.mean(projection)

                # 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산

                distances = projection - ensemble_mean
                
                # 각 앙상블 멤버의 거리를 저장
                ensemble_distances = {ens: distance for ens, distance in enumerate(distances)}

                


                fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{key_str}/{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/'
                ens_factor_uv=[]
                total_remove_uv = []
                u_mean_each = []
                v_mean_each = []
                
                base_output_path = os.path.join(
                pangu_dir, 
                'output_data', 
                'steering_wind',
                key_str, 
                f'{perturation_scale}ENS{surface_str}{upper_str}', 
                f'{min(ens_list)}_{max(ens_list)}_{start_str}_{target_str}'
            )


                # Paths for saving the arrays
                if steering_sign == 'y':
                    ens_factor_uv_path = os.path.join(base_output_path, 'ens_factor_uv')    #Steering 고도 한번에 고려 (2529, 2, 161, 241)
                    total_remove_uv_path = os.path.join(base_output_path, 'total_remove_uv')#각 고도 나눠서 생각 (2529, 2, 7, 161, 241)
                    u_mean_path = os.path.join(base_output_path, 'u_mean')
                    v_mean_path = os.path.join(base_output_path, 'v_mean')
                    
                    
                    ens_factor_uv = np.load(f'{ens_factor_uv_path}/{predict_interval}h{retro_opt}.npy')
                    total_remove_uv = np.load(f'{total_remove_uv_path}/{predict_interval}h{retro_opt}.npy')
                    
                    # u_mean = np.load(f'{u_mean_path}/{predict_interval}h{retro_opt}.npy')
                    # v_mean = np.load(f'{v_mean_path}/{predict_interval}h{retro_opt}.npy')
                
                    # if nearby_sign == 'y':
                    ens_factor_uv = np.array(ens_factor_uv)[group_idx]
                    u_mean.append(ens_factor_uv[:,0,:,:])
                    v_mean.append(ens_factor_uv[:,1,:,:])
                
                elif choosen_factor == 'u':
                    ens_factor_uv = ens_factor_uv[:,0,:,:]
                    total_remove_uv = total_remove_uv[:,0,:,:,:]
                    
                elif choosen_factor == 'v':
                    ens_factor_uv = ens_factor_uv[:,1,:,:]
                    total_remove_uv = total_remove_uv[:,1,:,:,:]
                    
                else:
                    ens_factor_uv=[]
                    for ens in np.array(ens_num_list)[group_idx]:
                        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                        ens_factor_uv.append(met.met_data(choosen_factor, level = altitude))
                    ens_factor_uv = np.array(ens_factor_uv)
                    non_uv_mean.append(ens_factor_uv)                
                

                # z_500 데이터를 numpy 배열로 변환
                ens_factor = np.copy(np.array(ens_factor_uv))

                if steering_sign == 'y':
                    ens_factor_quiver = (ens_factor[:,0,:,:], ens_factor[:,1,:,:])
                    cov_var_ratio = {}
                    
                    for ens_factor, uv_key in zip(ens_factor_quiver,['u','v']):
                        ens_factor_std = np.std(ens_factor, axis=0)
                        ens_factor = (ens_factor - np.mean(ens_factor, axis=0)) / ens_factor_std

                        # 공분산 및 분산 계산
                        cov_matrix = np.zeros_like(ens_factor[0])
                        var_matrix = np.zeros_like(ens_factor[0])

                        # for i in range(ens_factor.shape[1]):  # lat 방향
                        #     for j in range(ens_factor.shape[2]):  # lon 방향
                        #         cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
                        #         var_matrix[i, j] = np.var(ens_factor[:,i,j])

                    

                        # # NaN 또는 Inf 값을 0으로 대체
                        # cov_var_ratio[uv_key] = np.nan_to_num(cov_matrix / var_matrix * 111)
                        cov_var_ratio[uv_key] = compute_cov_var_ratio(distances, ens_factor.astype(np.float64))
                        
                        

                    dis = haversine_distance(lat_grid, lon_grid, np.ones_like(lat_grid)*mid_pos[0,1], np.ones_like(lat_grid)*mid_pos[0,0])
                    
                    us = []
                    vs = []
                    
                    for k in range(len(mid_pos)):
                        cov_var_inner['u'].append(np.mean(cov_var_ratio['u'][dis <= 333]))
                        cov_var_inner['v'].append(np.mean(cov_var_ratio['v'][dis <= 333]))
                        steer_quiver['u'].append(np.mean(ens_factor_uv[k,0,:,:][dis <= 333]))
                        steer_quiver['v'].append(np.mean(ens_factor_uv[k,1,:,:][dis <= 333]))
                        us.append(np.mean(ens_factor_uv[k,0,:,:][dis <= 333]))
                        vs.append(np.mean(ens_factor_uv[k,1,:,:][dis <= 333]))
                        cov_var_map['u'].append(cov_var_ratio['u'])
                        cov_var_map['v'].append(cov_var_ratio['v'])
                    
                    cov_var_inner_each['u'].append(np.mean(cov_var_ratio['u'][dis <= 333]))
                    cov_var_inner_each['v'].append(np.mean(cov_var_ratio['v'][dis <= 333])) 
                    steer_quiver_each['u'].append(np.mean(us))
                    steer_quiver_each['v'].append(np.mean(vs))
                               
                
                else:
                    ens_factor_std = np.std(ens_factor, axis=0)
                    ens_factor = (ens_factor - np.mean(ens_factor, axis=0)) / ens_factor_std

                    # 공분산 및 분산 계산
                    cov_matrix = np.zeros_like(ens_factor[0])
                    var_matrix = np.zeros_like(ens_factor[0])

                    for i in range(ens_factor.shape[1]):  # lat 방향
                        for j in range(ens_factor.shape[2]):  # lon 방향
                            cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
                            var_matrix[i, j] = np.var(ens_factor[:,i,j])

                    # NaN 또는 Inf 값을 0으로 대체
                    cov_var_ratio = np.nan_to_num(cov_matrix / var_matrix * 111)
                    for mp in mid_pos:
                        cov_var_nonuv.append(cov_var_ratio)

    cov_var_nearby[predict_interval] = {'inner': cov_var_inner, 
                                 'inner_each': cov_var_inner_each, 
                                 'map': cov_var_map}

#%%
            # steering wind를 그리기 위한 코드
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([130,145,25,35], crs=proj)
            gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12}
            gl.ylabel_style = {'size': 12}
        
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
            gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
            # gl.ylocator = mticker.MultipleLocator(5)
            ax.coastlines(zorder = 0)
            mid_pos_group = np.array(mid_pos_group)
            ax.scatter(mid_pos_group[:, 0], mid_pos_group[:, 1], alpha=1, s=2.5, c='green', zorder = 11)
        
            
            if steering_sign == 'y':

                u_mean_fig = np.mean(np.concatenate(u_mean, axis=0), axis=0)
                u_mean_fig = np.mean(np.concatenate(v_mean, axis=0), axis=0)
                # ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean_fig[::8,::8], u_mean_fig[::8,::8], length=5, linewidth=0.5, color = 'black')
                ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean_fig[::8,::8], u_mean_fig[::8,::8], length=5, linewidth=0.5, color = 'black')

                            
                # 태풍 영역 안에 들어오는 sensitivity의 방향 평균을 구하기, 정규화
                u_quiver = np.mean(np.array(cov_var_map['u']), axis=0)
                v_quiver = np.mean(np.array(cov_var_map['v']), axis=0)
                
                u_mean_inner = np.mean(cov_var_inner['u'])
                v_mean_inner = np.mean(cov_var_inner['v'])
                
                u_steer = np.mean(steer_quiver['u'])
                v_steer = np.mean(steer_quiver['v'])
                cov_var_total[predict_interval] = {'u': u_quiver,'v':v_quiver}
                
                qui_length = 500
                
                #Sensitivity를 파란색 화살표로 표시
                qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], u_quiver[::4, ::4], v_quiver[::4, ::4], scale=qui_length*20, color='blue', width=0.003)
                key = ax.quiverkey(qui, X=0.9, Y=0.97, U=qui_length, label=f'{qui_length}km', labelpos='E')
                
                # 태풍 영역에 들어오는 sensitivity의 방향 평균을 구하기
                inner_qui = ax.quiver(total_mean_lon, total_mean_lat, u_mean_inner, v_mean_inner, scale=qui_length*4, color='purple', width=0.003, zorder = 12)
                inner_key = ax.quiverkey(inner_qui, X=0.9, Y=0.93, U=qui_length/5, label=f'{int(qui_length/5)}km', labelpos='E')
            
                # dir_near[predict_interval]={'u': u_mean_inner, 'v': v_mean_inner}
                steer_qui = ax.quiver(total_mean_lon, total_mean_lat, u_steer, v_steer, scale=100, color='orange', width=0.003, zorder = 12)
                steer_key = ax.quiverkey(steer_qui, X=0.9, Y=0.89, U=5, label=f'5 m/s', labelpos='E')
            
            
                # 각 그룹의 길이를 계산합니다.
                lengths = [len(mem) for mem in group_mem_dict[predict_interval]]

                # 모든 그룹의 총 길이를 계산합니다.
                total_length = sum(lengths)

                # 각 그룹의 길이를 총 길이로 나누어 비율을 계산합니다.
                length_ratio = [length / total_length for length in lengths]
                # length_ratio = [len(mem) for mem in group_mem_dict[predict_interval]]/sum([len(mem) for mem in group_mem_dict[predict_interval]])
                for mid_pos_g, cov_var_u, cov_var_v, steer_u_each, steer_v_each,lr in zip(mid_pos_group, cov_var_inner_each['u'], cov_var_inner_each['v'], steer_quiver_each['u'],steer_quiver_each['v'],length_ratio):
                    # print(np.log1p(lr))
                    # lr = min(1, lr*2)
                    ax.quiver(mid_pos_g[0], mid_pos_g[1], cov_var_u, cov_var_v, scale=qui_length*4, color='brown', width=0.002, zorder = 11, alpha = 0.4)
                    # ax.quiver(mid_pos_g[0], mid_pos_g[1], steer_u_each, steer_v_each, scale=200, color='yellow', width=0.002, zorder = 11, alpha = 0.4)

                rect = patches.Rectangle((0.84, 0.87), 0.16, 0.15, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
                ax.add_patch(rect)
                
                u_mean_inner, v_mean_inner
                cov_u_dir, cov_v_dir = u_mean_inner/np.sqrt(u_mean_inner**2+v_mean_inner**2), v_mean_inner/np.sqrt(u_mean_inner**2+v_mean_inner**2)
                cov_proj = mid_pos_total @ np.array([cov_u_dir, cov_v_dir])
                
                
                correlation = np.corrcoef(mid_total_proj[:, 0], tar_total_proj[:, 0])[0, 1]
                correlations.append((predict_interval, correlation))
                
                correlation = np.corrcoef(cov_proj, tar_total_proj[:, 0])[0, 1]
                correlations_same.append((predict_interval, correlation))

            
            
            else:
                # print(np.shape((non_uv_mean)))
                non_uv_mean = np.concatenate(non_uv_mean, axis=0)
                # print(np.shape(non_uv_mean))
                # print(non_uv_mean[0]==non_uv_mean[1],non_uv_mean[0]==non_uv_mean[-1])
                # print(np.shape(non_uv_mean))
                non_uv_mean = np.mean(non_uv_mean, axis=0)
                # print(np.shape(cov_var_nonuv))
                # print(cov_var_nonuv[0] == cov_var_nonuv[1], cov_var_nonuv[0] == cov_var_nonuv[-1])
                cov_var_nonuv = np.mean(cov_var_nonuv, axis=0)
                contour = ax.contourf(lon_grid, lat_grid, cov_var_nonuv, cmap=pwp, levels=np.linspace(-200, 200, 17), transform=ccrs.PlateCarree())
                cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}) / Var({choosen_factor})', shrink = 1)
                cbar.locator = mticker.MultipleLocator(50)  # Set the colorbar ticks to have an interval of 0.5
                cbar.update_ticks()
                
                if choosen_factor == 'z':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,15001,60), colors='black')    
                elif choosen_factor == 't':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,401,2), colors='black')
                elif choosen_factor == 'q':
                    cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,21,1),    colors='black')
            
                cax.clabel()
                
            ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')

            # fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{start_str}_{target_str}_z/{choosen_factor}/'
            if not os.path.exists(f'{fig_dir}/{choosen_factor}/{altitude}hPa'):
                os.makedirs(f'{fig_dir}/{choosen_factor}/{altitude}hPa')
            else:
                pass
            
            plt.show()
            # if steering_sign == 'y':
            #     fig.savefig(f'{fig_dir}/{choosen_factor}/{predict_interval}h_{min_mem_threshold}.png',bbox_inches='tight')
            # else:
            #     fig.savefig(f'{fig_dir}/{choosen_factor}/{altitude}hPa/{predict_interval}h_{min_mem_threshold}.png', bbox_inches='tight')
            # plt.close()
            #%%
from matplotlib import font_manager
qui_length = 100
font_prop = font_manager.FontProperties(size=15)
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.set_extent([140,155,20,35], crs=proj)
ax.scatter(mid_pos_group[:, 0], mid_pos_group[:, 1], alpha=1, s=15, c='green', zorder = 11)
# ax.scatter(mid_pos[:,0], mid_pos[:,1], c='g', s=15, transform=proj)
ax.coastlines(resolution='10m', color='black', linewidth=0.5, zorder=10)
qui =ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4],u_quiver[::4,::4], v_quiver[::4,::4], color='blue',scale = 2000, width=0.002, transform=proj)
key = ax.quiverkey(qui, X=0.86, Y=0.98, U=qui_length, label=f'{qui_length} km', labelpos='E', zorder=22, fontproperties=font_prop)
inner_qui = ax.quiver(np.mean(mid_pos[:,0]), np.mean(mid_pos[:,1]), u_mean_inner, v_mean_inner, color='orange',scale = 2000, width=0.002, transform=proj)
inner_key = ax.quiverkey(inner_qui, X=0.86, Y=0.94, U=qui_length, label=f'{int(qui_length)} km', labelpos='E', zorder=22, fontproperties=font_prop)

gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))

mid_pos = np.array(mid_pos)
rect = patches.Rectangle((0.80, 0.92), 0.2, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
ax.add_patch(rect)
plt.show()
#%%
            if steering_sign == 'y':
                fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                # ax.set_extent([125,160,20,40], crs=proj)
                ax.set_extent([135,145,25,35], crs=proj)
                ax.coastlines()
                cvu = np.sum(np.array([cov_var_total[pi]['u'] for pi in cov_var_total]), axis = 0)
                cvv = np.sum(np.array([cov_var_total[pi]['v'] for pi in cov_var_total]), axis = 0)
                qui_length = 500
                
                #Sensitivity를 파란색 화살표로 표시
                qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], cvu[::4, ::4], cvv[::4, ::4], scale=qui_length*20, color='blue', width=0.003)
                key = ax.quiverkey(qui, X=0.9, Y=0.97, U=qui_length, label=f'{qui_length}km', labelpos='E')
                fig.savefig(f'{fig_dir}/{choosen_factor}/{predict_interval}h_sum.png',bbox_inches='tight')
                
            
            #Streamline으로 zoom in 지역 표시
            if steering_sign == 'y':
                fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([total_mean_lon-10, total_mean_lon+10, total_mean_lat-10, total_mean_lat+10], crs=proj)
                mid_pos_group = np.array(mid_pos_group)
                ax.scatter(mid_pos_group[:, 0], mid_pos_group[:, 1], alpha=1, s=2.5, c='green', zorder = 11)
                gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 12}
                gl.ylabel_style = {'size': 12}
            
                gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
                gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 5))
                ax.coastlines(zorder = 0)
                ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5, color = 'black')
                # ax.streamplot(lon_grid, lat_grid, u_mean, v_mean, density=1, linewidth=1, color='black', minlength=0.3, zorder = 0)
                            
                qui_length = 250
                
                #Sensitivity를 파란색 화살표로 표시
                qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], u_quiver[::4, ::4], v_quiver[::4, ::4], scale=qui_length*20, color='blue', width=0.003)
                key = ax.quiverkey(qui, X=0.865, Y=0.97, U=qui_length, label=f'{qui_length}km', labelpos='E')
                
                # 태풍 영역에 들어오는 sensitivity의 방향 평균을 구하기
                inner_qui = ax.quiver(total_mean_lon, total_mean_lat, u_mean_inner, v_mean_inner, scale=qui_length*4, color='purple', width=0.003, zorder = 12)
                inner_key = ax.quiverkey(inner_qui, X=0.865, Y=0.93, U=qui_length/5, label=f'{int(qui_length/5)}km', labelpos='E')
            
                steer_qui = ax.quiver(total_mean_lon, total_mean_lat, u_steer, v_steer, scale=100, color='orange', width=0.003, zorder = 12)
                steer_key = ax.quiverkey(steer_qui, X=0.865, Y=0.89, U=5, label=f'5 m/s', labelpos='E')
            
            
                # 각 그룹의 길이를 계산합니다.
                lengths = [len(mem) for mem in group_mem_dict[predict_interval]]

                # 모든 그룹의 총 길이를 계산합니다.
                total_length = sum(lengths)

                # 각 그룹의 길이를 총 길이로 나누어 비율을 계산합니다.
                length_ratio = [length / total_length for length in lengths]
                # length_ratio = [len(mem) for mem in group_mem_dict[predict_interval]]/sum([len(mem) for mem in group_mem_dict[predict_interval]])
                for mid_pos_g, cov_var_u, cov_var_v, steer_u_each, steer_v_each,lr in zip(mid_pos_group, cov_var_inner_each['u'], cov_var_inner_each['v'], steer_quiver_each['u'],steer_quiver_each['v'],length_ratio):
                    # print(np.log1p(lr))
                    # lr = min(1, lr*2)
                    ax.quiver(mid_pos_g[0], mid_pos_g[1], cov_var_u, cov_var_v, scale=qui_length*4, color='brown', width=0.002, zorder = 11, alpha = 0.4)
                    # ax.quiver(mid_pos_g[0], mid_pos_g[1], steer_u_each, steer_v_each, scale=200, color='yellow', width=0.002, zorder = 11, alpha = 0.4)

                rect = patches.Rectangle((0.8, 0.87), 0.2, 0.15, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
                ax.add_patch(rect)
                
                
                ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')

                # fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{start_str}_{target_str}_z/{choosen_factor}/'
                if not os.path.exists(f'{fig_dir}/{choosen_factor}/{altitude}hPa'):
                    os.makedirs(f'{fig_dir}/{choosen_factor}/{altitude}hPa')
                else:
                    pass
                
                fig.savefig(f'{fig_dir}/{choosen_factor}/{predict_interval}h_{min_mem_threshold}_zoom.png',bbox_inches='tight')
                plt.close()
            
            
            if steering_sign == 'y':
                fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([total_mean_lon-2, total_mean_lon+2, total_mean_lat-2, total_mean_lat+2], crs=proj)
                mid_pos_group = np.array(mid_pos_group)
                ax.scatter(mid_pos_group[:, 0], mid_pos_group[:, 1], alpha=1, s=20, c='green', zorder = 11)
                gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 12}
                gl.ylabel_style = {'size': 12}

            
                gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 2))
                gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 2))
                ax.coastlines(zorder = 0)
                # ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5, color = 'black')
                ax.streamplot(lon_grid, lat_grid, u_mean, v_mean, density=1, linewidth=1, color='black', minlength=0.3, zorder = 0)
                            
                qui_length = 100
                from matplotlib.font_manager import FontProperties
                font_props = FontProperties(size=15)  # 원하는 폰트 크기로 설정
                #Sensitivity를 파란색 화살표로 표시
                qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], u_quiver[::4, ::4], v_quiver[::4, ::4], scale=qui_length*20, color='blue', width=0.003)
                key = ax.quiverkey(qui, X=0.865, Y=0.97, U=qui_length, label=f'{qui_length}km', labelpos='E', fontproperties=font_props)
                
                # 태풍 영역에 들어오는 sensitivity의 방향 평균을 구하기
                inner_qui = ax.quiver(total_mean_lon, total_mean_lat, u_mean_inner, v_mean_inner, scale=qui_length*4, color='purple', width=0.003, zorder = 12)
                inner_key = ax.quiverkey(inner_qui, X=0.865, Y=0.93, U=int(qui_length/5), label=f'{int(qui_length/5)}km', labelpos='E', fontproperties=font_props)
            
                steer_qui = ax.quiver(total_mean_lon, total_mean_lat, u_steer, v_steer, scale=qui_length, color='orange', width=0.003, zorder = 12)
                steer_key = ax.quiverkey(steer_qui, X=0.865, Y=0.89, U=int(qui_length/20), label=f'{int(qui_length/20)} m/s', labelpos='E', fontproperties=font_props)
            
            
                # 각 그룹의 길이를 계산합니다.
                lengths = [len(mem) for mem in group_mem_dict[predict_interval]]

                # 모든 그룹의 총 길이를 계산합니다.
                total_length = sum(lengths)

                # 각 그룹의 길이를 총 길이로 나누어 비율을 계산합니다.
                length_ratio = [length / total_length for length in lengths]
                # length_ratio = [len(mem) for mem in group_mem_dict[predict_interval]]/sum([len(mem) for mem in group_mem_dict[predict_interval]])
                for mid_pos_g, cov_var_u, cov_var_v, steer_u_each, steer_v_each,lr in zip(mid_pos_group, cov_var_inner_each['u'], cov_var_inner_each['v'], steer_quiver_each['u'],steer_quiver_each['v'],length_ratio):
                    # print(np.log1p(lr))
                    # lr = min(1, lr*2)
                    ax.quiver(mid_pos_g[0], mid_pos_g[1], cov_var_u, cov_var_v, scale=qui_length*4, color='brown', width=0.002, zorder = 11, alpha = 0.6)
                    # ax.quiver(mid_pos_g[0], mid_pos_g[1], steer_u_each, steer_v_each, scale=200, color='yellow', width=0.002, zorder = 11, alpha = 0.4)

                rect = patches.Rectangle((0.8, 0.87), 0.2, 0.15, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
                ax.add_patch(rect)
                
                
                ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')

                # fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{start_str}_{target_str}_z/{choosen_factor}/'
                if not os.path.exists(f'{fig_dir}/{choosen_factor}/{altitude}hPa'):
                    os.makedirs(f'{fig_dir}/{choosen_factor}/{altitude}hPa')
                else:
                    pass
                
                fig.savefig(f'{fig_dir}/{choosen_factor}/{predict_interval}h_{min_mem_threshold}_zoom_more.png',bbox_inches='tight')
                plt.close()
            

length_list = list(map(sum, [[len(mem) for mem in group_mem_dict[key]] for key in group_mem_dict]))
plt.plot(group_mem_dict.keys(), length_list, marker='o')
plt.axhline(len(ens_num_list), color = 'red')
plt.xticks(np.arange(start_time_range,72+1,12))
plt.show()
plt.close()

#%%
u_mean

#%%
# 상관관계 시각화
if steering_sign == 'y':
    predict_intervals, corr_values = zip(*correlations)
    plt.plot(predict_intervals, np.abs(corr_values), marker='o', label = 'Spread')
    # predict_intervals, corr_values = zip(*correlations_opt)
    # plt.plot(predict_intervals, np.abs(corr_values), marker='x', label = 'Opt')
    predict_intervals, corr_values = zip(*correlations_same)
    plt.plot(predict_intervals, np.abs(corr_values), marker='^', label = 'Pos Cor')
    predict_intervals, corr_values = zip(*correlations_all)
    plt.plot(predict_intervals, np.abs(corr_values), marker='*', label = 'No Cor')
    
    plt.ylim(0, 1)  
    plt.xlim(start_time_range, 72)  
    plt.ylabel('Correlation')
    plt.xlabel('Predict time(h)')
    # plt.xlim(24, 120)  
    plt.grid(True)

    # x축 gridline 설정
    plt.xticks(np.arange(start_time_range, 72+1, 12))
    plt.legend(loc = 'lower right')
    plt.show()
    fig.savefig(f'{fig_dir}/Position_correlation_{start_str}_{target_str}.png',bbox_inches='tight')

#%%
#!초기 위치 어디인지 확인하기
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)          #분석 시작 시점
# start_time = datetime(2022,8,27,18)          #분석 시작 시점
target_time = datetime(2022,9,1,0)         #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")
time_range = int((target_time-key_time).total_seconds() / 3600)

# target_time 때도 살아있는 태풍만 추출
ens_num_list = []
for ens in ens_list:
    # if (target_time in ssv_dict[key_time][ens]) and (datetime(2022,8,27,12) in ssv_dict[key_time][ens]):
    # if target_time in ssv_dict[key_time][ens]:
    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
        ens_num_list.append(ens)
print(ens_num_list, len(ens_num_list))


fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
for predict_interval in np.arange(24,time_range+1,24):
# for predict_interval in np.arange(12,time_range+1,24):
# for predict_interval in np.arange(30,37,24):
# for predict_interval in [30]:
    datetime1 = key_time + timedelta(hours=int(predict_interval))
    
    mid_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
    mid_lat = [ssv_dict[key_time][ens][datetime1]['lat'] for ens in ens_num_list]
    tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]
    tar_lat = [ssv_dict[key_time][ens][target_time]['lat'] for ens in ens_num_list]

    # NumPy 배열로 변환
    mid_pos, tar_pos = np.array(mid_pos), np.array(tar_pos)

    # 경도의 왜곡을 보정
    corr_pos_tar = np.copy(tar_pos)
    corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca_tar = PCA(n_components=1)
    pca_tar.fit(corr_pos_tar)
    pca_tar.mean_[0] = pca_tar.mean_[0] / np.cos(np.radians(pca_tar.mean_[1])) + np.mean(tar_pos[:, 0])

    corr_pos_mid = np.copy(mid_pos)
    corr_pos_mid[:, 0] = (mid_pos[:, 0]-np.mean(mid_pos[:, 0])) * np.cos(np.radians(mid_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca_mid = PCA(n_components=1)
    pca_mid.fit(corr_pos_mid)
    pca_mid.mean_[0] = pca_mid.mean_[0] / np.cos(np.radians(pca_mid.mean_[1])) + np.mean(mid_pos[:, 0])


    # tc_pos와 pc_pos를 각각 pca_mid와 pca_tar에 사영
    # mid_proj = pca_mid.transform(corr_pos_mid)
    # tar_proj = pca_tar.transform(corr_pos_tar)
    # mid2tar = pca_tar.transform(corr_pos_mid)
    mid_proj = pca_mid.transform(mid_pos)
    tar_proj = pca_tar.transform(tar_pos)
    mid2tar = pca_tar.transform(mid_pos)
    mid_re = pca_mid.inverse_transform(mid_proj)
    tar_re = pca_tar.inverse_transform(tar_proj)
    tar2mid = pca_tar.inverse_transform(mid2tar)
    
    
    group1_idx = np.argsort(mid_proj[:,0])[:20]
    group2_idx = np.argsort(mid_proj[:,0])[-20:]
    group1 = mid_pos[group1_idx]
    group2 = mid_pos[group2_idx]
    
    
    setup_map(ax, back_color='y')
    ax.set_extent([135,152,22,32])
    # ax.set_extent([145,152,23,30])
    ax.gridlines(linestyle='--', linewidth=0.2)
    if predict_interval < 96:
        ax.scatter(group1[:,0], group1[:,1], c='red', label='Group1', alpha=0.2, marker='o')
        ax.scatter(group2[:,0], group2[:,1], c='blue', label='Group2', alpha = 0.2, marker='x')
    
    #! 어떤 시간대에서 최종 시간대에 대응하는 그룹을 그릴 것인지
    #// pos_24


    # for i in np.arange(24,time_range+1,12):
    # for i in np.arange(24, 24+1,24):
    for i in [24]:
    # for i in np.arange(18, 18+1,24):
        # datetime2 = key_time + timedelta(hours=int(i))
        pos_24 = [(ssv_dict[key_time][ens][key_time + timedelta(hours=int(i))]['lon'], ssv_dict[key_time][ens][key_time + timedelta(hours=int(i))]['lat']) for ens in ens_num_list]
        group1_24 = np.array(pos_24)[group1_idx]
        group2_24 = np.array(pos_24)[group2_idx]
    # pos_24 = [(ssv_dict[key_time][ens][key_time + timedelta(hours=int(36))]['lon'], ssv_dict[key_time][ens][key_time + timedelta(hours=int(36))]['lat']) for ens in ens_num_list]
    # group1_24 = np.array(pos_24)[group1_idx]
    # group2_24 = np.array(pos_24)[group2_idx]
        ret_pos = np.concatenate((group1_24, group2_24), axis=0)
        corr_pos_ret = np.copy(ret_pos)
        corr_pos_ret[:, 0] = (ret_pos[:, 0]-np.mean(ret_pos[:, 0])) * np.cos(np.radians(ret_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
        pca_ret = PCA(n_components=1)
        pca_ret.fit(corr_pos_ret)
        pca_ret.mean_[0] = pca_ret.mean_[0] / np.cos(np.radians(pca_ret.mean_[1])) + np.mean(ret_pos[:, 0])


        # if predict_interval == 108:
        # 대상이 되는 최종 시간대 지정
        if predict_interval == 120:
            print(1)
            # ax.quiver(pca_ret.mean_[0], pca_ret.mean_[1], pca_ret.components_[0, 0], pca_ret.components_[0, 1], scale=20, color='b', width=0.003, label='Principal Axis')
            # ax.quiver(pca_ret.mean_[0], pca_ret.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1], scale=20, color='r', width=0.003, label='Principal Axis')
            # ax.quiver(pca_ret.mean_[0], pca_ret.mean_[1], best_direction[0], best_direction[1], scale=20, color='b', width=0.003, label='Principal Axis')
            ax.scatter(group1_24[:,0], group1_24[:,1], c='yellow', label='Group1 24h', marker='o', alpha=0.2)
            ax.scatter(group2_24[:,0], group2_24[:,1], c='black', label='Group2 24h', marker='x', alpha=0.2)
            
            # Calculate the end position of the arrow
            # arrow_end_x = pca_ret.mean_[0] + pca_ret.components_[0, 0]  # Scale factor
            # arrow_end_y = pca_ret.mean_[1] + pca_ret.components_[0, 1]  # Scale factor
            arrow_end_x = pca_ret.mean_[0] + pca_ret.components_[0, 0]  # Scale factor
            arrow_end_y = pca_ret.mean_[1] + pca_ret.components_[0, 1]  # Scale factor

            # Format datetime2 as dd/hh
            # datetime_label = datetime2.strftime("%d/%H")

            # Add the datetime label at the end of the arrow
            # ax.text(
            #     arrow_end_x+1, arrow_end_y+1,
            #     datetime_label,
            #     fontsize=10, color='r', ha='right', va='bottom'
            # )
#%%
dir_all, dir_near

for key_time ,min_position in ssv_dict.items():
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    key_str = key_time.strftime("%m.%d.%HUTC")
    # ax.set_title(f'{key_time.strftime("%Y-%m-%d-%HUTC")} (+{predict_interval_list[-1]}h)', fontsize=20, loc = 'left')
    # ax.set_title(f'ENS{surface_str}{upper_str}{perturation_scale} Track\n{storm_name}', fontsize=20, loc = 'right')
    # ax.set_title(f'{storm_name}', fontsize=20, loc = 'right')
    ax.set_extent([120,155,20,45], crs=proj)
    setup_map(ax, back_color='n')

    # ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'Best track', transform=ax.projection, zorder=10)
    # model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=10)
    # cbar = plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
    # cbar.ax.tick_params(labelsize=15)

    
    # for i in range(len(storm_time)):
    #     new_time = storm_time[i].strftime("%Y/%m/%d/%HUTC")
    #     if new_time.endswith('00UTC'):
    #         dx, dy = 5, -0.5  # 시간 나타낼 위치 조정
    #         new_lon, new_lat = storm_lon[i] + dx, storm_lat[i] + dy
            
    #         # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
    #         ax.text(storm_lon[i], new_lat, new_time[8:-6]
    #                 , horizontalalignment='right', verticalalignment='top', fontsize=15, zorder = 20, fontweight = 'bold')



    # for ens in range(ens_num):  
    for ens in ens_list:  


        lons = [pos['lon'] for _,pos in min_position[ens].items()]
        lats = [pos['lat'] for _,pos in min_position[ens].items()]
        min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
        pred_times = [pos for pos,_ in min_position[ens].items()]
        # print(ens)
        lc = ax.plot(lons, lats, alpha=0.01, color = 'orange')

        #? 시간 표시 00UTC만 표시, 없앨듯

    # 각 ens의 lons와 lats 리스트를 저장할 리스트
    all_lons = []
    all_lats = []
    
    from itertools import zip_longest
    
    for ens in ens_list:
        lons = [pos['lon'] for _, pos in min_position[ens].items()]
        lats = [pos['lat'] for _, pos in min_position[ens].items()]

        all_lons.append(lons)
        all_lats.append(lats)

    # 요소별 평균 계산
    # zip_longest를 사용하여 가장 긴 리스트 길이에 맞춰 각 요소끼리 묶어줍니다.
    # fillvalue=np.nan으로 설정하여 빈 요소는 NaN으로 처리합니다.
    avg_lons = [np.nanmean(x) for x in zip_longest(*all_lons, fillvalue=np.nan)]
    avg_lats = [np.nanmean(x) for x in zip_longest(*all_lats, fillvalue=np.nan)]

    qui_length = 500
    # qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], u_quiver[::4, ::4], v_quiver[::4, ::4], scale=qui_length*20, color='blue', width=0.003)
    # key = ax.quiverkey(qui, X=0.9, Y=0.97, U=qui_length, label=f'{qui_length}km', labelpos='E')
    
    # 태풍 영역에 들어오는 sensitivity의 방향 평균을 구하기
    u_ens_sen = np.array([dir_all[i]['u'] for i in dir_all])
    v_ens_sen = np.array([dir_all[i]['v'] for i in dir_all])
    inner_qui = ax.quiver(avg_lons[:17], avg_lats[:17], u_ens_sen, v_ens_sen, scale=qui_length*4, color='grey', width=0.003, zorder = 12)
    inner_key = ax.quiverkey(inner_qui, X=0.92, Y=0.97, U=qui_length/5, label=f'{int(qui_length/5)}km', labelpos='E')

    u_ens_sen = np.array([dir_near[i]['u'] for i in dir_near])
    v_ens_sen = np.array([dir_near[i]['v'] for i in dir_near])
    inner_qui = ax.quiver(avg_lons[:17], avg_lats[:17], u_ens_sen, v_ens_sen, scale=qui_length*4, color='purple', width=0.003, zorder = 12)
    inner_key = ax.quiverkey(inner_qui, X=0.92, Y=0.93, U=qui_length/5, label=f'{int(qui_length/5)}km', labelpos='E')
            
        
    # rect = patches.Rectangle((0.8, 0.87), 0.2, 0.15, linewidth=1, edgecolor='black', facecolor='white', transform=ax.transAxes)
    # ax.add_patch(rect)
    # ax.legend(loc='upper right')

    # lons_all = np.concatenate([np.array([pos['lon'] for _, pos in min_position[ens].items()]) for ens in ens_list])
    # lats_all = np.concatenate([np.array([pos['lat'] for _, pos in min_position[ens].items()]) for ens in ens_list])


    # xy = np.vstack([lons_all, lats_all])
    # kde = gaussian_kde(xy)
    # positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    # f = np.reshape(kde(positions).T, lon_grid.shape)


    # levels = np.linspace(0.0005, 0.015, 100)
    # cf = ax.contourf(lon_grid, lat_grid, f, levels=levels, transform=proj, cmap='jet')
    plt.show()
    # fig.savefig(f'{pangu_dir}/plot/Ensemble_track_{key_str}.png',bbox_inches='tight')

#%%
all_lons = []
all_lats = []

from itertools import zip_longest

for ens in ens_list:
    lons = [pos['lon'] for _, pos in min_position[ens].items()]
    lats = [pos['lat'] for _, pos in min_position[ens].items()]

    all_lons.append(lons)
    all_lats.append(lats)

# 요소별 평균 계산
# zip_longest를 사용하여 가장 긴 리스트 길이에 맞춰 각 요소끼리 묶어줍니다.
# fillvalue=np.nan으로 설정하여 빈 요소는 NaN으로 처리합니다.
avg_lons = [np.nanmean(x) for x in zip_longest(*all_lons, fillvalue=np.nan)]
avg_lats = [np.nanmean(x) for x in zip_longest(*all_lats, fillvalue=np.nan)]

avg_lons

#%%
#! 태풍 근처 배경장 확인하기
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
nearby_sign = 'y'                           #가까운 태풍만 추출
distance_threshold = 0                      #가까운 태풍의 거리
steering_sign = 'n'                         #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
steer_uni_alt = 0                           #steering wind를 구할 때, 고도를 하나
# choosen_factor_list = ['z','t','q']       #구하고자 하는 변수
choosen_factor_list = ['z']         #구하고자 하는 변수
# altitude_list = [1000,850,700,500,300,200]  #각 변수에 대해 구하고자 하는 고도
altitude_list = [850,500,250]               #각 변수에 대해 구하고자 하는 고도
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x
axis_opt = 'quiver'                         #axis 뭘로 잡을지, opt: 위치 상관관계 최대인 axis, tar: 최종 위치의 axis, mid: 중간 위치의 axis, lon: 경도, lat: 위도
data_sign = 'n'                             #기존의 데이터를 사용할 것인지, n이면 새로 구함

if nearby_sign == 'y':
    nearby_sign_name = '_nearby'
else:
    nearby_sign_name = ''
    

# steering_sign이 y면 steering wind에 대해서만 구하기
if steering_sign == 'y':
    altitude_list = ['850_200']
    choosen_factor_list = ['steering_wind']
    

ens_num_list = []
for ens in ens_list:
    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
        ens_num_list.append(ens)


# target_time 때도 살아있는 태풍만 추출
correlations = []
correlations_tar = []
correlations_opt = []
correlations_gg = []
correlations_df = []
ens_num_list = []

# for predict_interval in np.arange(start_time_range,total_time_range+1,6):
# for predict_interval in np.arange(54,total_time_range+1,6):
for predict_interval in np.arange(start_time_range,72+1,6):
# for predict_interval in np.arange(24,73,12):
# for predict_interval in np.arange(72,109,12):
# for predict_interval in np.arange(120,124,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))

    ens_num_list = []
    for ens in ens_list:
        if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
            ens_num_list.append(ens)
    
    # 데이터를 추출합니다
    ens_pos = [(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]

    # 위도(lat) 기준으로 데이터를 정렬합니다
    pos_sorted_by_lat = sorted(ens_pos, key=lambda x: x[2])  # x[2]는 위도를 나타냅니다

    # 위도가 가장 낮은 10개와 가장 높은 10개를 추출합니다
    group1 = pos_sorted_by_lat[:50]  # 가장 낮은 10개
    group2 = pos_sorted_by_lat[-50:]  # 가장 높은 10개
    group3 = pos_sorted_by_lat[len(pos_sorted_by_lat) // 2 - 5:len(pos_sorted_by_lat) // 2 + 5]
    
    # group1과 group2에 있는 ens 번호만 추출합니다
    group1 = [item[0] for item in group1]
    group2 = [item[0] for item in group2]
    group3 = [item[0] for item in group3]
    
    mid_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
    tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]
    gg_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in group1 + group2]
    gg_tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in group1 + group2]
    
    # NumPy 배열로 변환
    mid_pos, tar_pos, gg_pos, gg_tar_pos = np.array(mid_pos), np.array(tar_pos), np.array(gg_pos), np.array(gg_tar_pos)
    
   

    
    uv_all_alt = {}
    uv_all_1 = {}
    uv_all_2 = {}
    for choosen_factor in choosen_factor_list:
        uv_all_alt[choosen_factor] = []
        uv_all_1[choosen_factor] = []
        uv_all_2[choosen_factor] = []
        for altitude in altitude_list:
            fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{key_str}/{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/'
            print(choosen_factor, altitude)     
            ens_factor_uv=[]
            total_remove_uv = []
            u_mean_each = []
            v_mean_each = []
            
            base_output_path = os.path.join(
                pangu_dir, 
                'output_data', 
                'steering_wind',
                key_str, 
                f'{perturation_scale}ENS{surface_str}{upper_str}', 
                f'{min(ens_list)}_{max(ens_list)}_{start_str}_{target_str}'
            )

            # Paths for saving the arrays
            ens_factor_uv_path = os.path.join(base_output_path, 'ens_factor_uv')
            total_remove_uv_path = os.path.join(base_output_path, 'total_remove_uv')
            u_mean_path = os.path.join(base_output_path, 'u_mean')
            v_mean_path = os.path.join(base_output_path, 'v_mean')

            ens_factor_uv = np.load(f'{ens_factor_uv_path}/{predict_interval}h{retro_opt}.npy')
            total_remove_uv = np.load(f'{total_remove_uv_path}/{predict_interval}h{retro_opt}.npy')
            
            u_mean = np.load(f'{u_mean_path}/{predict_interval}h{retro_opt}.npy')
            v_mean = np.load(f'{v_mean_path}/{predict_interval}h{retro_opt}.npy')
            
            
            center_lon = np.mean(mid_pos[:,0])
            center_lat = np.mean(mid_pos[:,1])
            u_mean = np.mean(ens_factor_uv[:,0,:,:], axis=0)
            v_mean = np.mean(ens_factor_uv[:,1,:,:], axis=0)
            
            u_mean_s = np.zeros_like(u_mean)
            u_mean_s[:] = np.nan  # NaN으로 초기화하여 이후 평균 계산을 쉽게 함
            v_mean_s = np.zeros_like(v_mean)
            v_mean_s[:] = np.nan  # NaN으로 초기화
            count = np.zeros_like(u_mean)  # 각 위치에 대한 값의 개수를 추적

            for s_lon, s_lat, s_u, s_v in zip(mid_pos[:, 0], mid_pos[:, 1], ens_factor_uv[:,0,:,:], ens_factor_uv[:,1,:,:]):
                dis = haversine_distance(lat_grid, lon_grid, np.ones_like(lat_grid) * s_lat, np.ones_like(lat_grid) * s_lon)
                s_u = np.mean(s_u[dis <= 333])
                s_v = np.mean(s_v[dis <= 333])
                s_idx = np.where((lat_grid == s_lat) & (lon_grid == s_lon))

                if count[s_idx] > 0:  # 이미 값이 존재하는 경우 가중 평균 계산
                    u_mean_s[s_idx] = (u_mean_s[s_idx] * count[s_idx] + s_u) / (count[s_idx] + 1)
                    v_mean_s[s_idx] = (v_mean_s[s_idx] * count[s_idx] + s_v) / (count[s_idx] + 1)
                else:  # 처음 값을 넣는 경우
                    u_mean_s[s_idx] = s_u
                    v_mean_s[s_idx] = s_v

                count[s_idx] += 1  # 해당 위치에 대한 값 개수 증가

            # g1_u,  g2_u, g1_v, g2_v = [], [], [], []
            # for s_lon, s_lat, s_u, s_v in zip(mid_pos[:, 0], mid_pos[:, 1], ens_factor_uv[:,0,:,:], ens_factor_uv[:,1,:,:]):
            #     dis = haversine_distance(lat_grid, lon_grid, np.ones_like(lat_grid) * s_lat, np.ones_like(lat_grid) * s_lon)
            #     s_u = np.mean(s_u[dis <= 333])
            #     s_v = np.mean(s_v[dis <= 333])
            #     s_idx = np.where((lat_grid == s_lat) & (lon_grid == s_lon))

            #     if count[s_idx] > 0:  # 이미 값이 존재하는 경우 가중 평균 계산
            #         u_mean_s[s_idx] = (u_mean_s[s_idx] * count[s_idx] + s_u) / (count[s_idx] + 1)
            #         v_mean_s[s_idx] = (v_mean_s[s_idx] * count[s_idx] + s_v) / (count[s_idx] + 1)
            #     else:  # 처음 값을 넣는 경우
            #         u_mean_s[s_idx] = s_u
            #         v_mean_s[s_idx] = s_v

            #     count[s_idx] += 1  # 해당 위치에 대한 값 개수 증가

            from scipy.interpolate import griddata
            
            # u_mean_s = np.where(count < 2 ,np.nan, u_mean_s)
            # v_mean_s = np.where(count < 2 ,np.nan, v_mean_s)
            valid_mask = ~np.isnan(u_mean_s)

            # NaN이 아닌 값들의 좌표를 추출합니다.
            points = np.array([lat_grid[valid_mask], lon_grid[valid_mask]]).T

            # NaN이 아닌 값들을 추출합니다.
            values_u = u_mean_s[valid_mask]
            values_v = v_mean_s[valid_mask]

            # NaN 값이 있는 위치를 찾습니다.
            nan_mask = np.isnan(u_mean_s)
            grid_points = np.array([lat_grid[nan_mask], lon_grid[nan_mask]]).T

            # griddata를 사용하여 외삽을 수행합니다.
            u_mean_s[nan_mask] = griddata(points, values_u, grid_points, method='linear')
            v_mean_s[nan_mask] = griddata(points, values_v, grid_points, method='linear')
            
            
            fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{key_str}/{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/'
            if not os.path.exists(f'{fig_dir}/u_steer/'):
                os.makedirs(f'{fig_dir}/u_steer/')
                os.makedirs(f'{fig_dir}/v_steer/')
            else:
                pass
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([center_lon-2, center_lon+2, center_lat-2, center_lat+2], crs=proj)
            setup_map(ax, back_color='n')
            cf = ax.contourf(lon_grid, lat_grid, u_mean_s, cmap='gist_ncar', levels=np.linspace(-10, 0, 41), transform=ccrs.PlateCarree())
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=1, s=2.5, c='black', zorder = 11)
            cbar = plt.colorbar(cf, ax = ax, shrink = 0.8)
            cbar.set_label("$m/s$", fontsize=12)
            cbar.set_ticks(np.arange(-10, 0.1, 1))
            fig.savefig(f'{fig_dir}/u_steer/{predict_interval}h', bbox_inches='tight')
            plt.close()
            
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([center_lon-2, center_lon+2, center_lat-2, center_lat+2], crs=proj)
            setup_map(ax, back_color='n')
            cf = ax.contourf(lon_grid, lat_grid, v_mean_s, cmap='gist_ncar', levels=np.linspace(-5, 5, 41), transform=ccrs.PlateCarree())
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=1, s=2.5, c='black', zorder = 11)
            cbar = plt.colorbar(cf, ax = ax, shrink = 0.8)
            cbar.set_label("$m/s$", fontsize=12)
            cbar.set_ticks(np.arange(-5, 5.1, 1))
            fig.savefig(f'{fig_dir}/v_steer/{predict_interval}h', bbox_inches='tight')
            plt.close()
            
            
            