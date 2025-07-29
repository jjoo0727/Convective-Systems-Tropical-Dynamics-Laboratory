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
from ty_pkg import latlon_extent, storm_info, haversine_distance, Met, calculate_bearing_position, tc_finder, WindFieldSolver

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
new_ssv = 'n'          #새로 생성할 것인지 여부, n이면 기존 파일 불러옴
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
                                        min_position[ens], mask_size = 2.5, init_size=5, local_min_size = 5, mslp_z_dis = 250, wind_thres=0)

            
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
                                                min_position[ens], mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 1000, wind_thres=0)
                else:
                    min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                                wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                                min_position[ens], mask_size = 2.5, local_min_size = 5, back_prop='y', mslp_z_dis = 250, wind_thres=0)
                    
                
                min_position[ens] = {k: min_position[ens][k] for k in sorted(min_position[ens])}
                
        ssv_dict[ssv_key] = min_position

    # with open(rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict{retro_opt}_{min(ens_list)}_{max(ens_list)}.pkl', 'wb') as f:
    #     pickle.dump(ssv_dict, f)

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

        
        with open(rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict{retro_opt}_{min(ens_list)}_{max(ens_list)}.pkl', 'rb') as f:
            ssv_dict = pickle.load(f)
        
#%%
#! 가장 상관관계 높은 axis 구하기, 2nd main code
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,12)          #분석 시작 시점
target_time = datetime(2022,9,1,0)          #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")

total_time_range = int((target_time - key_time).total_seconds() / 3600)
start_time_range = int((start_time  - key_time).total_seconds() / 3600)

# 변수 지정
nearby_sign = 'n'                           #가까운 태풍만 추출
distance_threshold = 0.25                   #가까운 태풍의 거리
tc_remove_sign = 'y'                        #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x
axis_opt = 'opt'                            #axis 뭘로 잡을지, opt: 위치 상관관계 최대인 axis, tar: 최종 위치의 axis, mid: 중간 위치의 axis, lon: 경도, lat: 위도
data_sign = 'n'                             #기존의 데이터를 사용할 것인지, n이면 새로 구함
save_sign = 'y'                             #구한 데이터를 저장할 것인지     

if nearby_sign == 'y':
    nearby_sign_name = '_nearby'
else:
    nearby_sign_name = ''
    

# tc_remove_sign이 y면 steering wind에 대해서만 구하기
if tc_remove_sign == 'y':
    altitude_list = ['850_200']
    choosen_factor_list = ['steering_wind']
else:
    axis_opt = 'tar'  


# target_time 때도 살아있는 태풍만 추출
ens_num_list = []
for ens in ens_list:
    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
        ens_num_list.append(ens)
print(ens_num_list, len(ens_num_list))



correlations = []
correlations_tar = []
correlations_opt = []
correlations_df = []



# for predict_interval in np.arange(start_time_range,total_time_range+1,6):
for predict_interval in np.arange(total_time_range,total_time_range+1,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))

    mid_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
    tar_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]
    
    # NumPy 배열로 변환
    mid_pos, tar_pos = np.array(mid_pos), np.array(tar_pos)
    
    
    if nearby_sign == 'y':
        # 각 위치 간의 거리 계산
        from scipy.spatial.distance import cdist
        distances = cdist(mid_pos, mid_pos)
        distances

        # 0.5도 이내에 있는 모든 위치 찾기
        close_pairs = np.where(distances <= distance_threshold)

        # 밀집 그룹 찾기
        from collections import defaultdict
        groups = defaultdict(list)
        for i, j in zip(*close_pairs):
            if i != j:
                groups[i].append(ens_num_list[j])

        # 가장 큰 그룹의 ens 리스트 출력
        max_group = max(groups.items(), key=lambda x: len(x[1]))[1]
        max_group = list(set(max_group))  # 중복 제거
        max_group_idx = np.array([ens_num_list.index(b) for b in max_group if b in ens_num_list])
        tar_pos = tar_pos[max_group_idx]
        mid_pos = mid_pos[max_group_idx]
        print(len(ens_num_list), len(max_group))
    
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
    
    
    projection = pca_tar.transform(corr_pos_tar)[:, 0]  # 주축에 투영된 데이터 (1차원)
    principal_component = pca_tar.components_[0]

    # 투영된 데이터의 ensemble mean 계산
    ensemble_mean = np.mean(projection)

    # 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산
    distances = projection - ensemble_mean

    # 각 앙상블 멤버의 거리를 저장
    ensemble_distances = {ens: distance for ens, distance in enumerate(distances)}

    
    mid_proj = pca_mid.transform(mid_pos)
    tar_proj = pca_tar.transform(tar_pos)
    mid2tar = pca_tar.transform(mid_pos)
    mid_re = pca_mid.inverse_transform(mid_proj)
    tar_re = pca_tar.inverse_transform(tar_proj)
    tar2mid = pca_tar.inverse_transform(mid2tar)

    

    
    for choosen_factor in choosen_factor_list:
        for altitude in altitude_list:
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
            
            if data_sign == 'n':
                for ens in ens_num_list:
                    print(ens)
                    center_lon, center_lat = ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']
                    
                    output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
                    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                    
                    if choosen_factor != 'steering_wind':
                        ens_factor_uv.append(met.met_data(choosen_factor, level = altitude))
                    else:
                        u_list = []
                        v_list = []
                        
                        
                        # if len(steer_pres) > 1:
                        dis_cal_sign = 'y'
                        final_mask = None
                        for steer_altitude in steer_pres:
                            
                            div = met.divergence(level = steer_altitude)
                            vort = met.vorticity(level = steer_altitude)
                            vort_850 = met.vorticity(level = 850)
                            
                            if tc_remove_sign == 'y':
                                ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850, dis_cal_sign = dis_cal_sign, final_mask = final_mask)
                                dis_cal_sign = 'n'
                                u_ty, v_ty, final_mask = ty_wind.solve()
                                u_list.append(met.met_data('u', level = steer_altitude)-u_ty)
                                v_list.append(met.met_data('v', level = steer_altitude)-v_ty)

                            else:
                                u_list.append(met.met_data('u', level = steer_altitude))
                                v_list.append(met.met_data('v', level = steer_altitude))
                                
                        u,v = np.zeros(np.shape(u_list[0])), np.zeros(np.shape(u_list[0]))
                        
                    
                        for i in range(len(steer_pres)-1):
                            u += (u_list[i]+u_list[i+1])/2*(steer_pres[i]-steer_pres[i+1])
                            v += (v_list[i]+v_list[i+1])/2*(steer_pres[i]-steer_pres[i+1])
                        
                        u/=np.ptp(steer_pres)
                        v/=np.ptp(steer_pres)
                            
                        # else:
                        #     div = met.divergence(level = steer_pres[0])
                        #     vort = met.vorticity(level = steer_pres[0])
                            
                        #     if tc_remove_sign == 'y':
                        #         ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850)
                        #         u_ty, v_ty = ty_wind.solve()
                        #         u = met.met_data('u', level = steer_pres[0]-u_ty)
                        #         v = met.met_data('v', level = steer_pres[0]-v_ty)
                                
                        #     else:
                        #         u = met.met_data('u', level = steer_pres[0])
                        #         v = met.met_data('v', level = steer_pres[0])
                        
                        u_mean_each.append(u)
                        v_mean_each.append(v)
                        ens_factor_uv.append([u,v])
                        total_remove_uv.append([u_list, v_list])
                        
                        fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
                        ax.quiver(lon_grid[::4,::4], lat_grid[::4,::4], u[::4,::4], v[::4,::4])
                        ax.coastlines()
                        plt.show()


                        if ens == 0:
                            u0 = u
                            v0 = v
                            
                u_mean = np.mean(np.array(u_mean_each), axis=0)
                v_mean = np.mean(np.array(v_mean_each), axis=0)
            
                # Ensure the figure directory exists
                # fig_path = os.path.join(fig_dir, f'{altitude}hPa')
                # os.makedirs(fig_path, exist_ok=True)

                # Base path for output data
#%%

                # Ensure all required directories exist
                if save_sign == 'y':
                    os.makedirs(ens_factor_uv_path, exist_ok=True)
                    os.makedirs(total_remove_uv_path, exist_ok=True)
                    os.makedirs(u_mean_path, exist_ok=True)
                    os.makedirs(v_mean_path, exist_ok=True)
                    
                    np.save(f'{ens_factor_uv_path}/{predict_interval}h{retro_opt}.npy', np.array(ens_factor_uv))
                    np.save(f'{total_remove_uv_path}/{predict_interval}h{retro_opt}.npy', np.array(total_remove_uv))
                    np.save(f'{u_mean_path}/{predict_interval}h{retro_opt}.npy', u_mean)
                    np.save(f'{v_mean_path}/{predict_interval}h{retro_opt}.npy', v_mean)
            

           