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
predict_interval_list = np.arange(0,24*7+1,6)  
ens_num = 1000
new_ssv = 'n'          #새로 생성할 것인지 여부, n이면 기존 파일 불러옴

        
#! 태풍 경로 정보 새로 생성하기
if new_ssv == 'y':
    for first_str in ['2022/08/27/00UTC']:
        first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
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
                
                #mask_size는 다음 태풍 찾을 때 그 위경도 안이 아니면 안 찾음S
                #처음 시작할 때는 5도 이내에만 들어오면 되고
                #mslp_z_dis는 250이 기본인데 이는 mslp 지역 최솟값과 z지역 최댓값이 250km 이내여야 pass
                #mslp_2hpa는 주변 8방위로 2hPa이 감소하는지 보는건데 일단은 'n'으로 걸어둠
                #아래에서는 mask_size말고는 아무런 제약을 안 걸었음
                min_position[ens] = tc_finder(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                                        wind_speed, predict_time, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                                        min_position[ens], mask_size = 2.5, init_size=5, local_min_size = 5)


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

    with open(rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict.pkl', 'wb') as f:
        pickle.dump(ssv_dict, f)

else:
    for first_str in ['2022/08/27/00UTC']:
        first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
        ssv_key = first_time
        surface_factors.sort()
        upper_factors.sort()
        surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
        upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가
        
        datetime_list = np.array([first_time + timedelta(hours=int(hours)) for hours in predict_interval_list])
        # datetime_array = np.array([(first_time + timedelta(hours=int(hours))) for hours in predict_interval_list])
        storm_lon, storm_lat, storm_mslp, storm_time = storm_info(pangu_dir, storm_name, storm_year, datetime_list = datetime_list, wind_thres=0)   #태풍 영문명, 년도 입력

        
        with open(rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict.pkl', 'rb') as f:
            ssv_dict = pickle.load(f)
        



#%%
#! 가장 상관관계 높은 axis 구하기, 2nd main code
# 시간 지정
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,27,12)          #분석 시작 시점
target_time = datetime(2022,9,1,0)          #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")

total_time_range = int((target_time - key_time).total_seconds() / 3600)
start_time_range = int((start_time  - key_time).total_seconds() / 3600)

# 변수 지정
nearby_sign = 'y'                           #가까운 태풍만 추출
distance_threshold = 0.25                   #가까운 태풍의 거리
steering_sign = 'n'                         #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
# choosen_factor_list = ['z','t','q']       #구하고자 하는 변수
choosen_factor_list = ['u','v']             #구하고자 하는 변수
altitude_list = [1000,850,700,500,300,200]  #각 변수에 대해 구하고자 하는 고도
# altitude_list = [850,500,200]             #각 변수에 대해 구하고자 하는 고도
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x
axis_opt = 'uv'                                 #axis 뭘로 잡을지, opt: 위치 상관관계 최대인 axis, tar: 최종 위치의 axis, mid: 중간 위치의 axis, lon: 경도, lat: 위도
data_sign = 'y'                             #기존의 데이터를 사용할 것인지, n이면 새로 구함
save_sign = 'n'                             #!구한 데이터를 저장할 것인지, 만드는 코드 아니면 n으로 하자     

if nearby_sign == 'y':
    nearby_sign_name = '_nearby'
else:
    nearby_sign_name = ''
    

# steering_sign이 y면 steering wind에 대해서만 구하기
if steering_sign == 'y':
    altitude_list = ['850_200']
    choosen_factor_list = ['steering_wind']



# target_time 때도 살아있는 태풍만 추출
ens_num_list = []
for ens in range(ens_num):
    if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
        ens_num_list.append(ens)
print(ens_num_list, len(ens_num_list))



correlations = []
correlations_tar = []
correlations_opt = []
correlations_df = []



# for predict_interval in np.arange(start_time_range,total_time_range+1,6):
# for predict_interval in np.arange(54,total_time_range+1,6):
for predict_interval in np.arange(start_time_range,48+1,6):
# for predict_interval in np.arange(48,49,6):
# for predict_interval in np.arange(120,124,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))

    # 데이터를 추출합니다
    ens_pos = [(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]

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

    
    uv_all_alt = {}
    uv_all_one = {}
    uv_all_1 = {}
    uv_all_2 = {}
    for choosen_factor in choosen_factor_list:
        uv_all_alt[choosen_factor] = []
        uv_all_one[choosen_factor] = []
        uv_all_1[choosen_factor] = []
        uv_all_2[choosen_factor] = []
        
        for altitude in altitude_list:
            print(choosen_factor, altitude)     
            ens_factor_uv=[]
            total_remove_uv = []
            u_mean_each = []
            v_mean_each = []
            
            base_output_path = os.path.join(
                pangu_dir, 
                'output_data', 
                first_str, 
                f'{perturation_scale}ENS{surface_str}{upper_str}', 
                f'{ens_num}_{start_str}_{target_str}'
            )

            # Paths for saving the arrays
            ens_factor_uv_path = os.path.join(base_output_path, 'ens_factor_uv')
            total_remove_uv_path = os.path.join(base_output_path, 'total_remove_uv')
            u_mean_path = os.path.join(base_output_path, 'u_mean')
            v_mean_path = os.path.join(base_output_path, 'v_mean')
            
            if data_sign == 'n':
                for ens in ens_num_list:
                    # print(ens)
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


                # Ensure all required directories exist
                if save_sign == 'y':
                    os.makedirs(ens_factor_uv_path, exist_ok=True)
                    os.makedirs(total_remove_uv_path, exist_ok=True)
                    os.makedirs(u_mean_path, exist_ok=True)
                    os.makedirs(v_mean_path, exist_ok=True)
                    
                    np.save(f'{ens_factor_uv_path}/{predict_interval}h.npy', np.array(ens_factor_uv))
                    np.save(f'{total_remove_uv_path}/{predict_interval}h.npy', np.array(total_remove_uv))
                    np.save(f'{u_mean_path}/{predict_interval}h.npy', u_mean)
                    np.save(f'{v_mean_path}/{predict_interval}h.npy', v_mean)
            
            else:
                ens_factor_uv = np.load(f'{ens_factor_uv_path}/{predict_interval}h.npy')
                total_remove_uv = np.load(f'{total_remove_uv_path}/{predict_interval}h.npy')
                # print(np.shape(total_remove_uv))
                u_mean = np.load(f'{u_mean_path}/{predict_interval}h.npy')
                v_mean = np.load(f'{v_mean_path}/{predict_interval}h.npy')
                ens_factor_uv2 = np.copy(ens_factor_uv)
                
                if choosen_factor == 'u':
                    ens_factor_uv = ens_factor_uv[:,0,:,:]
                    total_remove_uv = total_remove_uv[:,0,:,:,:]
                elif choosen_factor == 'v':
                    ens_factor_uv = ens_factor_uv[:,1,:,:]
                    total_remove_uv = total_remove_uv[:,1,:,:,:]
            
            if nearby_sign == 'y':
                ens_factor_uv = np.array(ens_factor_uv)[max_group_idx]
                
                if steering_sign == 'y':
                    u_mean = np.mean(ens_factor_uv[:,0,:,:], axis=0)
                    v_mean = np.mean(ens_factor_uv[:,1,:,:], axis=0)
            
            
            non_uv_mean = np.mean(ens_factor_uv, axis=0)
            
                # v_mean = v_mean[max_group_idx]
                
                     
            
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
            if not os.path.exists(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}'):
                os.makedirs(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}')
            else:
                pass
            plt.savefig(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}/{predict_interval}h.png')
            plt.close()


            # z_500 데이터를 numpy 배열로 변환
        #     ens_factor = np.array(ens_factor_uv)
            
        #     if steering_sign == 'y':
        #         if axis_opt == 'opt':
        #             ens_factor = ens_factor[:,0,:,:]*best_direction[0] + ens_factor[:,1,:,:]*best_direction[1]
        #         elif axis_opt == 'tar':
        #             ens_factor = ens_factor[:,0,:,:]*pca_tar.components_[0, 0] + ens_factor[:,1,:,:]*pca_tar.components_[0, 1]
        #         elif axis_opt == 'mid':
        #             ens_factor = ens_factor[:,0,:,:]*pca_mid.components_[0, 0] + ens_factor[:,1,:,:]*pca_mid.components_[0, 1]
        #         elif axis_opt == 'lon':
        #             ens_factor = ens_factor[:,0,:,:]
        #         elif axis_opt == 'lat':
        #             ens_factor = ens_factor[:,1,:,:]
            
        #     ens_factor_std = np.std(ens_factor, axis=0)
        #     ens_factor = (ens_factor - np.mean(ens_factor, axis=0)) / ens_factor_std

        #     # 공분산 및 분산 계산
        #     cov_matrix = np.zeros_like(ens_factor[0])
        #     var_matrix = np.zeros_like(ens_factor[0])

        #     for i in range(ens_factor.shape[1]):  # lat 방향
        #         for j in range(ens_factor.shape[2]):  # lon 방향
        #             cov_matrix[i, j] = np.cov(distances, ens_factor[:, i, j])[0, 1]
        #             var_matrix[i, j] = np.var(ens_factor[:,i,j])

        
        #    # 공분산/분산 비율 계산
        #     cov_var_ratio = cov_matrix / var_matrix * 111 #! 이거 한번 확인할 필요 있음!


        #     # NaN 또는 Inf 값을 0으로 대체
        #     cov_var_ratio = np.nan_to_num(cov_var_ratio)

        #     # 지도에 결과 표시
        #     fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        #     contour = ax.contourf(lon_grid, lat_grid, cov_var_ratio, cmap=pwp, levels=np.linspace(-400, 400, 17), transform=ccrs.PlateCarree())
        #     cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}) / Var({choosen_factor})', shrink = 0.6)
        #     cbar.locator = ticker.MultipleLocator(100)  # Set the colorbar ticks to have an interval of 0.5
        #     cbar.update_ticks()
        #     ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=0.7, s=0.6, c='green')
        #     setup_map(ax, back_color='n')
            
            
        #     if steering_sign == 'y':
        #         ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5)

                        
        #         if axis_opt == 'opt':
        #             ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], best_direction[0], best_direction[1], scale=20, color='b', width=0.003, label='Principal Axis')
        #         elif axis_opt == 'tar':
        #             ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1], scale=20, color='r', width=0.003, label='Principal Axis')
        #         elif axis_opt == 'mid':
        #             ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_mid.components_[0, 0], pca_mid.components_[0, 1], scale=20, color='g', width=0.003, label='Principal Axis')
        #         elif axis_opt == 'lon':
        #             ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 1, 0, scale=20, color='black', width=0.003, label='Principal Axis')
        #         elif axis_opt == 'lat':
        #             ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 0, 1, scale=20, color='black', width=0.003, label='Principal Axis')
            
        #     else:
        #         if choosen_factor == 'z':
        #             cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,15001,60), colors='black')
        #         elif choosen_factor == 't':
        #             cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(200,401,5), colors='black')
        #         elif choosen_factor == 'q':
        #             cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(0,21,1),    colors='black')
        #         elif choosen_factor == 'u':
        #             cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(-100,101,10), colors='black')
        #         elif choosen_factor == 'v':
        #             cax = ax.contour(lon_grid, lat_grid, non_uv_mean, levels=np.arange(-100,101,10), colors='black')
                
        #         cax.clabel()
            
                
                
        #     ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')
        #     fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/{choosen_factor}/'
        #     if not os.path.exists(f'{fig_dir}/{altitude}hPa'):
        #         os.makedirs(f'{fig_dir}/{altitude}hPa')
        #     else:
        #         pass
        #     if steering_sign == 'y':
        #         fig.savefig(f'{fig_dir}/{predict_interval}h.png',bbox_inches='tight')
        #     else: 
        #         fig.savefig(f'{fig_dir}/{altitude}hPa/{predict_interval}h.png', bbox_inches='tight')
        #     plt.close()
            
        if choosen_factor == 'u' or choosen_factor == 'v':
            total_remove_uv = np.swapaxes(total_remove_uv, 0, 1)
            
            for uv_uni in total_remove_uv:
                uv_uni_alt  = []

                uv_uni_1  = []
                uv_uni_2  = []

                for mp, uv, ens_uv in zip(mid_pos, uv_uni, ens_num_list):
                    # idx = np.where((lat_grid == mp[1]) & (lon_grid == mp[0])) # 오직 태풍 중심 위치 바람 정보 가져오기
                    #태풍 범위 내의 u 또는 v 평균 내기
                    dis_uv = haversine_distance(lat_grid, lon_grid, np.ones_like(lat_grid)*mp[1], np.ones_like(lat_grid)*mp[0])
                    # print(uv.shape)
                    # print(uv[dis_uv <= 333].shape)
                    
                    uv_uni_alt.append(np.mean(uv[dis_uv <= 333])) #태풍 중심으로부터 333km 이내의 모든 바람 데이터를 가져와서 평균
                    if ens_uv in group1:
                        uv_uni_1.append(np.mean(uv[dis_uv <= 333]))
                    if ens_uv in group2:
                        uv_uni_2.append(np.mean(uv[dis_uv <= 333]))
                    
                uv_uni_one = np.array(uv_uni_alt)           #순서 바꾸면 큰일남
                uv_uni_alt = np.mean(np.array(uv_uni_alt), axis = 0)
                uv_uni_1 = np.mean(np.array(uv_uni_1), axis = 0)
                uv_uni_2 = np.mean(np.array(uv_uni_2), axis = 0)
                uv_all_alt[choosen_factor].append(uv_uni_alt)
                uv_all_one[choosen_factor].append(uv_uni_one)
                uv_all_1[choosen_factor].append(uv_uni_1)
                uv_all_2[choosen_factor].append(uv_uni_2)
        
        
        uv_all_alt[choosen_factor] = np.array(uv_all_alt[choosen_factor])
        uv_all_one[choosen_factor] = np.array(uv_all_one[choosen_factor])
        uv_all_1[choosen_factor] = np.array(uv_all_1[choosen_factor])
        uv_all_2[choosen_factor] = np.array(uv_all_2[choosen_factor])
    
        uv = np.zeros_like(uv_all_one[choosen_factor][0])
        for i in range(len(steer_pres)-1):
            uv += (uv_all_one[choosen_factor][i]+uv_all_one[choosen_factor][i+1])/2*(steer_pres[i]-steer_pres[i+1])
        uv/=np.ptp(steer_pres)
        
        uv_all_one[choosen_factor] = uv
    
    
    if 'u' in choosen_factor_list and 'v' in choosen_factor_list:
        
        uv_data = np.stack((uv_all_one['u'], uv_all_one['v']), axis=-1)
        pca_uv = PCA(n_components=1)
        pca_uv.fit_transform(uv_data)
        
        # plt.plot(uv_all_alt['u']*best_direction[0] + uv_all_alt['v']*best_direction[1], steer_pres,marker='o', label = 'opt_mean', color = 'orange')
        plt.plot(uv_all_1['u']*best_direction[0] + uv_all_1['v']*best_direction[1]-(uv_all_alt['u']*best_direction[0] + uv_all_alt['v']*best_direction[1]), steer_pres,marker = 'x', label = 'opt_group1', alpha = 0.5, color = 'orange')
        plt.plot(uv_all_2['u']*best_direction[0] + uv_all_2['v']*best_direction[1]-(uv_all_alt['u']*best_direction[0] + uv_all_alt['v']*best_direction[1]), steer_pres,marker = '^', label = 'opt_group2', alpha = 0.5, color = 'orange')
        
        # plt.plot(uv_all_alt['u']*pca_tar.components_[0, 0] + uv_all_alt['v']*pca_tar.components_[0, 1], steer_pres,marker='o', label = 'tar_mean', color = 'green')
        plt.plot(uv_all_1['u']*pca_tar.components_[0, 0] + uv_all_1['v']*pca_tar.components_[0, 1]-(uv_all_alt['u']*pca_tar.components_[0, 0] + uv_all_alt['v']*pca_tar.components_[0, 1]), steer_pres,marker = 'x', label = 'tar_group1', alpha = 0.5, color = 'green')
        plt.plot(uv_all_2['u']*pca_tar.components_[0, 0] + uv_all_2['v']*pca_tar.components_[0, 1]-(uv_all_alt['u']*pca_tar.components_[0, 0] + uv_all_alt['v']*pca_tar.components_[0, 1]), steer_pres,marker = '^', label = 'tar_group2', alpha = 0.5, color = 'green')
        
        plt.plot(uv_all_1['u']*pca_uv.components_[0, 0] + uv_all_1['v']*pca_uv.components_[0, 1]-(uv_all_alt['u']*pca_uv.components_[0, 0] + uv_all_alt['v']*pca_uv.components_[0, 1]), steer_pres,marker = 'x', label = 'uv_group1', alpha = 0.5, color = 'black')
        plt.plot(uv_all_2['u']*pca_uv.components_[0, 0] + uv_all_2['v']*pca_uv.components_[0, 1]-(uv_all_alt['u']*pca_uv.components_[0, 0] + uv_all_alt['v']*pca_uv.components_[0, 1]), steer_pres,marker = '^', label = 'uv_group2', alpha = 0.5, color = 'black')
        
        # plt.plot(uv_all_alt['u'], steer_pres, marker='o', label = 'u_mean', color = 'blue')
        # plt.plot(uv_all_1['u'], steer_pres, marker='x', label = 'u_group1', alpha = 0.5, color = 'blue')
        # plt.plot(uv_all_2['u'], steer_pres, marker='^', label = 'u_group2', alpha = 0.5, color = 'blue')
        # plt.plot(uv_all_alt['v'], steer_pres, marker='o', label = 'v', color = 'red')
        # plt.plot(uv_all_1['v'], steer_pres, marker='x', label = 'v_group1', alpha = 0.5, color = 'red')
        # plt.plot(uv_all_2['v'], steer_pres, marker='^', label = 'v_group2', alpha = 0.5, color = 'red')
        
        plt.title(f'{predict_interval}h', fontweight = 'bold')
        
        plt.xlim(-2,2)
        plt.axvline(0, color = 'black')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid()
        plt.xlabel('Wind speed($m/s$)')
        plt.ylabel('Pressure(hPa)')
        plt.show()       
        
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)  # 극 좌표계 사용

        # 원형 그래프에 데이터 플로팅
        ax.plot(angles, dir_corr, label='Correlation', linewidth=2)  # 상관관계 선 그리기
        # 최대 상관관계 위치에 화살표 표시
        max_corr_radian = angles[max_index]
        
        # ax.quiver(max_corr_radian, 0, 0, best_correlation, angles='xy', scale_units='xy', scale=1, color='red', label='Max Correlation')
        ax.quiver(max_corr_radian, -1, 0, best_correlation + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='Max Corr Direction')
        # PCA 방향에 화살표 표시
        pca_value = dir_corr[int(rad_pca /np.pi /2 * len(angles))]
        ax.quiver(rad_pca, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='green', label='PCA Direction')
        rad_uv = np.arctan2(pca_uv.components_[0, 1], pca_uv.components_[0, 0])
        pca_value = dir_corr[int(rad_uv /np.pi /2 * len(angles))]
        ax.quiver(rad_uv, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, label=f'uv Direction')
        # pca_value = dir_corr[int(rad_df /np.pi /2 * len(angles))]
        # ax.quiver(rad_df, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='PCA Direction')
        ax.axvline(rad_tar, color='red', linewidth=2, label = 'Target Axis Direction')
        
        ax.set_ylim(-1, 1)
        ax.set_theta_zero_location('E')  # 0도를 북쪽(위)으로 설정
        ax.set_yticks(np.arange(-1, 1.1, 0.5))

        # 저장 및 보여주기
        if not os.path.exists(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}'):
            os.makedirs(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}')
        else:
            pass
        plt.legend(loc = 'lower left')
        plt.show()
        
        
        
        if nearby_sign == 'y':
            ens_factor_uv2 = np.array(ens_factor_uv2)[max_group_idx]
        ens_factor = np.array(ens_factor_uv2)
            
            
        if axis_opt == 'opt':
            ens_factor = ens_factor[:,0,:,:]*best_direction[0] + ens_factor[:,1,:,:]*best_direction[1]
        elif axis_opt == 'tar':
            ens_factor = ens_factor[:,0,:,:]*pca_tar.components_[0, 0] + ens_factor[:,1,:,:]*pca_tar.components_[0, 1]
        elif axis_opt == 'mid':
            ens_factor = ens_factor[:,0,:,:]*pca_mid.components_[0, 0] + ens_factor[:,1,:,:]*pca_mid.components_[0, 1]
        elif axis_opt == 'uv':
            ens_factor = ens_factor[:,0,:,:]*pca_uv.components_[0, 0] + ens_factor[:,1,:,:]*pca_uv.components_[0, 1]
            
        elif axis_opt == 'lon':
            ens_factor = ens_factor[:,0,:,:]
        elif axis_opt == 'lat':
            ens_factor = ens_factor[:,1,:,:]
        
        
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
        cov_var_ratio = cov_matrix / var_matrix * 111 #! 이거 한번 확인할 필요 있음!


        # NaN 또는 Inf 값을 0으로 대체
        cov_var_ratio = np.nan_to_num(cov_var_ratio)

        # 지도에 결과 표시
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        contour = ax.contourf(lon_grid, lat_grid, cov_var_ratio, cmap=pwp, levels=np.linspace(-400, 400, 17), transform=ccrs.PlateCarree())
        cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}) / Var({choosen_factor})', shrink = 0.6)
        cbar.locator = ticker.MultipleLocator(100)  # Set the colorbar ticks to have an interval of 0.5
        cbar.update_ticks()
        ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=0.7, s=0.6, c='green')
        setup_map(ax, back_color='n')
        
        
        ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5)

                
        if axis_opt == 'opt':
            ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], best_direction[0], best_direction[1], scale=20, color='b', width=0.003, label='Principal Axis')
        elif axis_opt == 'tar':
            ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1], scale=20, color='r', width=0.003, label='Principal Axis')
        elif axis_opt == 'mid':
            ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_mid.components_[0, 0], pca_mid.components_[0, 1], scale=20, color='g', width=0.003, label='Principal Axis')
        elif axis_opt == 'uv':
            ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_uv.components_[0, 0], pca_uv.components_[0, 1], scale=20, color='black', width=0.003, label='Principal Axis')
        elif axis_opt == 'lon':
            ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 1, 0, scale=20, color='black', width=0.003, label='Principal Axis')
        elif axis_opt == 'lat':
            ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 0, 1, scale=20, color='black', width=0.003, label='Principal Axis')
        
        
            
            
        ax.quiver(pca_tar.mean_[0], pca_tar.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1]  , scale=20, color='r', width=0.003, label='Principal Axis')
        fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/{choosen_factor}/'
        if not os.path.exists(f'{fig_dir}/{altitude}hPa'):
            os.makedirs(f'{fig_dir}/{altitude}hPa')
        else:
            pass
        
        fig.savefig(f'{fig_dir}/{predict_interval}h.png',bbox_inches='tight')
        plt.show()
        plt.close()
    
#%%

# 상관관계 시각화
predict_intervals, corr_values = zip(*correlations)
plt.plot(predict_intervals, np.abs(corr_values), marker='o', label = 'Projected Each axis')
# predict_intervals, corr_values = zip(*correlations_tar)
# plt.plot(predict_intervals, np.abs(corr_values), marker='^', label = 'Projection_fianl_axis')
predict_intervals, corr_values = zip(*correlations_opt)
plt.plot(predict_intervals, np.abs(corr_values), marker='x', label = 'Projected Optimized axis')
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
fig.savefig(f'{pangu_dir}/plot/Sensitivity/Position_correlation_{start_str}_{target_str}.png',bbox_inches='tight')


