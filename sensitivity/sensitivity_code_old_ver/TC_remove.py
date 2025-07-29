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

        
        with open(rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/ssv_dict{retro_opt}_{min(ens_list)}_{max(ens_list)}.pkl', 'rb') as f:
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
    for ens in ens_list:  


        lons = [pos['lon'] for _,pos in min_position[ens].items()]
        lats = [pos['lat'] for _,pos in min_position[ens].items()]
        min_values = [pos['mslp'] for _,pos in min_position[ens].items()]
        pred_times = [pos for pos,_ in min_position[ens].items()]
        # print(ens)
        lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)

        #? 시간 표시 00UTC만 표시, 없앨듯

        if ens == 0:
            lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)
            ax.scatter(lons, lats, c='red', linewidth=2, alpha=1, zorder=10, label = 'No perturbation')

            # for i in range(len(pred_times)):
            #     if pred_times[i].hour == 0:
            #         ax.text(lons[i],lats[i], str(pred_times[i].day)
            #             , horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder = 6)

        
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
    fig.savefig(f'{pangu_dir}/plot/Ensemble_track_{key_str}.png',bbox_inches='tight')
    
#%%
# storm_time.index(datetime(2022,8,28,12))
output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
            
            
for predict_interval in predict_interval_list:
    predict_time = first_time + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    mslp = met.met_data('MSLP')
    z = met.met_data('z', level = 'all')
#%%
dis_dict = {}
# key_time = datetime(2022,8,28, 12)
for e, data_total in ssv_dict[datetime(2022,8,27,0)].items():
    
    dis_list = []
    for key_time, data in data_total.items():
        time_delta = (key_time - datetime(2022,8,27,0)).total_seconds() / 3600
        if time_delta not in dis_dict:
            dis_dict[time_delta] = []
        lon = data['lon']
        lat = data['lat']
        if key_time not in storm_time:
            continue
        
        ty_lon, ty_lat = storm_lon[np.where(storm_time == datetime(2022,8,28,12))][0], storm_lat[np.where(storm_time == datetime(2022,8,28,12))][0]
        
        d = haversine_distance(lat, lon, ty_lat, ty_lon)
        dis_list.append(d)
    dis_dict[time_delta].append(np.mean(dis_list))

dis_dict
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
mean_dis_dict = {k: v for k, v in dis_dict.items() if k in df['Time Delta'].unique()}
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
#! main figure code
key_time = datetime(2022,8,27,0)        #처음 시점 지정
target_time = datetime(2022,9,1,0)      #위치 projection을 구하고자 하는 시간
key_str = key_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")
time_range = int((target_time-key_time).total_seconds() / 3600)


steering_sign = 'y'                        #태풍 제거를 진행할 것인지를 판단
choosen_factor_list = ['z','t','q','u','v']
altitude_list = [1000,850,700,500,300,200]  #각 변수에 대해 구하고자 하는 고도
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x

# steering_sign이 y면 steering wind에 대해서만 구하기
if steering_sign == 'y':
    altitude_list = ['850_200']
    choosen_factor_list = ['steering_wind']

# target_time 때도 살아있는 태풍만 추출
ens_num_list = []
for ens in range(ens_num):
    # if (target_time in ssv_dict[key_time][ens]) and (key_time in ssv_dict[key_time][ens]):
    if (target_time in ssv_dict[key_time][ens]) and (datetime(2022,8,27,12) in ssv_dict[key_time][ens]):
    # if target_time in ssv_dict[key_time][ens]:
        ens_num_list.append(ens)
print(ens_num_list, f'{ens_num - len(ens_num_list)} missing')
    
# for datetime1 in [datetime(2022,8,28,0,0),datetime(2022,8,29,0,0),datetime(2022,8,30,0,0),datetime(2022,8,31,0,0), datetime(2022,9,1,0,0)]:
# for predict_interval in np.arange(24,time_range+1,6):
for predict_interval in np.arange(12,24,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))
    print(datetime1)
    

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
    tc_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in ens_num_list]
    pc_pos = [(ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]
    
    # NumPy 배열로 변환
    tc_pos, pc_pos = np.array(tc_pos), np.array(pc_pos)
    # 경도의 왜곡을 보정
    corr_pos = np.copy(pc_pos)
    corr_pos[:, 0] = (pc_pos[:, 0]-np.mean(pc_pos[:, 0])) * np.cos(np.radians(pc_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca = PCA(n_components=1)
    pca.fit(corr_pos)
    pca.mean_[0] = pca.mean_[0] / np.cos(np.radians(pca.mean_[1])) + np.mean(pc_pos[:, 0])


    # 데이터를 주축에 투영
    projection = pca.transform(corr_pos)[:, 0]  # 주축에 투영된 데이터 (1차원)
    principal_component = pca.components_[0]

    # 투영된 데이터의 ensemble mean 계산
    ensemble_mean = np.mean(projection)

    # 각 앙상블 멤버의 투영 데이터와 ensemble mean 사이의 거리 계산
    distances = projection - ensemble_mean

    # 각 앙상블 멤버의 거리를 저장
    ensemble_distances = {ens: distance for ens, distance in enumerate(distances)}


    for choosen_factor in choosen_factor_list:
        for altitude in altitude_list:     
            ens_factor=[]
            u_mean = []
            v_mean = []
            
            for ens in ens_num_list:
                center_lon, center_lat = ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']
                
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
                            vort_850 = met.vorticity(level = 850)
                            
                            if steering_sign == 'y':
                                ty_wind = WindFieldSolver(lat_grid, lon_grid, center_lat, center_lon, vort, div, vort_850)
                                u_ty, v_ty = ty_wind.solve()
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
            cov_var_ratio = cov_matrix / var_matrix * 111 #! 이거 한번 확인할 필요 있음!


            # NaN 또는 Inf 값을 0으로 대체
            cov_var_ratio = np.nan_to_num(cov_var_ratio)

            fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            contour = ax.contourf(lon_grid, lat_grid, cov_var_ratio, cmap=pwp, levels=np.linspace(-400, 400, 21), transform=ccrs.PlateCarree())
            cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}_{altitude}) / Var({choosen_factor}_{altitude})', shrink = 0.6)
            cbar.locator = ticker.MultipleLocator(100)  # Set the colorbar ticks to have an interval of 0.5
            cbar.update_ticks()
            ax.scatter(tc_pos[:, 0], tc_pos[:, 1], alpha=0.7, s=0.6, c='green')
            setup_map(ax, back_color='n')
            # ax.set_extent([100,160,5,45])
            # plt.plot(np.mean(tc_loc, axis=0)[0], np.mean(tc_loc, axis=0)[1], marker = tcmarkers.TS, color = 'red')
            # plt.title(f'{storm_name}\n{choosen_factor.capitalize()} {altitude}hPa', loc = 'right')
            # plt.title(f'{key_str}(+{predict_interval}h)\n{target_time.strftime("%m.%d.%HUTC")}', loc = 'left')
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
            fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{storm_name}_{key_str}_{target_str}/{choosen_factor}/'
            if not os.path.exists(f'{fig_dir}/{altitude}hPa'):
                os.makedirs(f'{fig_dir}/{altitude}hPa')
            else:
                pass
            if steering_sign == 'y':
                fig.savefig(f'{fig_dir}/{predict_interval}h.png',bbox_inches='tight')
            else: 
                fig.savefig(f'{fig_dir}/{altitude}hPa/{predict_interval}h.png', bbox_inches='tight')
            plt.close()


#%%
#! 위치가 영향을 주는가 확인용 코드, 2개의 그룹 각각의 위치 표시하는 코드
key_time = datetime(2022,8,27,0)        #처음 시점 지정
target_time = datetime(2022,9,1,0)      #위치 projection을 구하고자 하는 시간
key_str = key_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")


ens_num_list = []
for ens in range(ens_num):
    # if (target_time in ssv_dict[key_time][ens]) and (key_time in ssv_dict[key_time][ens]):
    # if (target_time in ssv_dict[key_time][ens]) and (datetime(2022,8,27,12) in ssv_dict[key_time][ens]):
    if target_time in ssv_dict[key_time][ens]:
        ens_num_list.append(ens)

# 데이터를 추출합니다
# ens_pos = np.array([(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list])


# # 위도(lat) 기준으로 데이터를 정렬합니다
# pos_sorted_by_lat = sorted(ens_pos, key=lambda x: x[2])  # x[2]는 위도를 나타냅니다

# # 위도가 가장 낮은 10개와 가장 높은 10개를 추출합니다
# group1 = pos_sorted_by_lat[:10]  # 가장 낮은 10개
# group2 = pos_sorted_by_lat[-10:]  # 가장 높은 10개
# group3 = pos_sorted_by_lat[len(pos_sorted_by_lat) // 2 - 5:len(pos_sorted_by_lat) // 2 + 5]


# # group1과 group2에 있는 ens 번호만 추출합니다
# group1 = np.array([int(item[0]) for item in group1])
# group2 = np.array([int(item[0]) for item in group2])
# group3 = np.array([int(item[0]) for item in group3])
#%%
#!초기 위치 어디인지 확인하기
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)          #분석 시작 시점
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
# for predict_interval in np.arange(36,97,24):
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
    
    
    group1_idx = np.argsort(mid_proj[:,0])[:10]
    group2_idx = np.argsort(mid_proj[:,0])[-10:]
    group1 = mid_pos[group1_idx]
    group2 = mid_pos[group2_idx]
    
    
    setup_map(ax, back_color='y')
    ax.set_extent([135,152,22,32])
    ax.gridlines(linestyle='--', linewidth=0.2)
    ax.scatter(group1[:,0], group1[:,1], c='red', label='Group1', alpha=0.2, marker='o')
    ax.scatter(group2[:,0], group2[:,1], c='blue', label='Group2', alpha = 0.2, marker='x')
    
    for i in np.arange(24,time_range+1,24):
    # for i in np.arange(12, 12+1,24):
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
# predict_interval=36
for predict_interval in np.arange(24,49,6):
    g1_mean = []
    g1_u_mean = []
    g1_v_mean = []
    
    for ens in group1:
        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)      
        g1_mean.append(met.met_data('z',level=500))
        # g1_mean.append(met.vorticity(level=850))
        g1 = met.met_data('z', level=500)
        g1_u = met.met_data('u', level=850)
        g1_v = met.met_data('v', level=850)
        g1_u_mean.append(g1_u)
        g1_v_mean.append(g1_v)
        
        
    g2_mean = [] 
    g2_u_mean = []
    g2_v_mean = []
    for ens in group2:
        output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)      
        g2_mean.append(met.met_data('z',level=500))
        # g2_mean.append(met.vorticity(level=850))
        g2= met.met_data('z', level=500)
        g2_u = met.met_data('u', level=850)
        g2_v = met.met_data('v', level=850)
        g2_u_mean.append(g2_u)
        g2_v_mean.append(g2_v)

    g1_mean = np.mean(np.array(g1_mean), axis=0)
    g1_u_mean = np.mean(np.array(g1_u_mean), axis=0)
    g1_v_mean = np.mean(np.array(g1_v_mean), axis=0)
    g2_mean = np.mean(np.array(g2_mean), axis=0)
    g2_u_mean = np.mean(np.array(g2_u_mean), axis=0)
    g2_v_mean = np.mean(np.array(g2_v_mean), axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    setup_map(ax, back_color='n')
    # plt.scatter(lon_grid, lat_grid, c=g2_mean-g1_mean, cmap='coolwarm')
    ax.scatter(pos_sorted_by_lat[:10][:,1], pos_sorted_by_lat[:10][:,2])
    # plt.barbs(lon_grid[::4,::4], lat_grid[::4,::4], g1_u_mean[::4,::4]-g2_u_mean[::4,::4],g1_v_mean[::4,::4]-g2_v_mean[::4,::4], length=3)
    # plt.barbs(lon_grid[::4,::4], lat_grid[::4,::4], g1_u[::4,::4]-g2_u[::4,::4],g1_v[::4,::4]-g2_v[::4,::4], length=5)
    # plt.scatter(lon_grid, lat_grid, c=g2-g1, cmap='coolwarm', vmin=-10, vmax=10)
    # c1 = plt.contour(lon_grid, lat_grid, g1_mean, colors='red', levels=np.linspace(5400, 6000, 31))
    # c2 = plt.contour(lon_grid, lat_grid, g2_mean, colors='blue', levels=np.linspace(5400, 6000, 31))
    # c1.clabel(fmt='%d')
    # c2.clabel(fmt='%d')
    # plt.colorbar(orientation='horizontal')
    plt.title(f'{predict_interval}h')
    plt.show()
#%%
pos_sorted_by_lat[:10][1,:]
#%%
ssv_dict.keys()
#%%

for key_time ,min_position in ssv_dict.items():
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    # ax.set_title(f'{key_time.strftime("%Y-%m-%d-%HUTC")} (+{predict_interval_list[-1]}h)', fontsize=20, loc = 'left')
    # ax.set_title(f'ENS{surface_str}{upper_str}{perturation_scale} Track\n{storm_name}', fontsize=20, loc = 'right')
    # ax.set_title(f'{storm_name}', fontsize=20, loc = 'right')
    ax.set_extent([140,150,20,30], crs=proj)
    setup_map(ax)

    # ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'Best track', transform=ax.projection, zorder=3)
    # model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=3)
    # plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
    for ens in group1:
    # for ens in range(34,35):
    # for ens in group2:

        lons = [pos['lon'] for _,pos in min_position[ens].items() if _.hour == 0]
        lats = [pos['lat'] for _,pos in min_position[ens].items() if _.hour == 0]
        min_values = [pos['mslp'] for _,pos in min_position[ens].items() if _.hour == 0]
        pred_times = [pos for pos,_ in min_position[ens].items() if pos.hour == 0]
        #         lons = [pos['lon'] for _,pos in min_position[ens].items() if _.hour == 12]
        # lats = [pos['lat'] for _,pos in min_position[ens].items() if _.hour == 12]
        # min_values = [pos['mslp'] for _,pos in min_position[ens].items() if _.hour == 12] 
        # pred_times = [pos for pos,_ in min_position[ens].items() if pos.hour == 12]
        # print(ens)
        # lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('Reds'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)
        lc = ax.scatter(lons, lats, c=min_values, cmap=plt.get_cmap('Reds'), s=60, alpha=0.2)

        # for i in range(len(pred_times)):
        #     if pred_times[i].hour == 0:
        #         ax.text(lons[i],lats[i], str(pred_times[i].day)
        #             , horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder = 6)

    for ens in group2:
    # for ens in range(34,35):
    # for ens in group2:

        lons = [pos['lon'] for _,pos in min_position[ens].items() if _.hour == 0]
        lats = [pos['lat'] for _,pos in min_position[ens].items() if _.hour == 0]
        min_values = [pos['mslp'] for _,pos in min_position[ens].items() if _.hour == 0] 
        pred_times = [pos for pos,_ in min_position[ens].items() if pos.hour == 0]
        #         lons = [pos['lon'] for _,pos in min_position[ens].items() if _.hour == 12]
        # lats = [pos['lat'] for _,pos in min_position[ens].items() if _.hour == 12]
        # min_values = [pos['mslp'] for _,pos in min_position[ens].items() if _.hour == 12] 
        # pred_times = [pos for pos,_ in min_position[ens].items() if pos.hour == 12]
        # print(ens)
        # lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('Blues'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1)
        lc = ax.scatter(lons, lats, c=min_values, cmap=plt.get_cmap('Blues'), s=100, alpha=0.2, marker='x')
        
        # for i in range(len(pred_times)):
        #     if pred_times[i].hour == 0:
        #         ax.text(lons[i],lats[i], str(pred_times[i].day)
        #             , horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder = 6)


        
    # ax.legend(loc='upper right')


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
#!위치 상관관계 구하기
key_time = datetime(2022,8,27,0)        #처음 시점 지정
start_time = datetime(2022,8,28,0)
target_time = datetime(2022,9,1,0)      #위치 projection을 구하고자 하는 시간
key_str = key_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")
time_range = int((target_time-key_time).total_seconds() / 3600)

    
# target_time 때도 살아있는 태풍만 추출
ens_num_list = []
for ens in ens_list:
    # if (target_time in ssv_dict[key_time][ens]) and (datetime(2022,8,27,12) in ssv_dict[key_time][ens]):
    if target_time in ssv_dict[key_time][ens] and start_time in ssv_dict[key_time][ens]:
        ens_num_list.append(ens)
print(ens_num_list, len(ens_num_list))
    
ens_pos = [(ens, ssv_dict[key_time][ens][target_time]['lon'], ssv_dict[key_time][ens][target_time]['lat']) for ens in ens_num_list]
# ens_pos
correlations = []
correlations_mid2tar = []
correlations_mid_lat = []
correlations_mid_tar_lat = []
mid2tar_projections = []
correlations_opt = []
for predict_interval in np.arange(24,time_range+1,6):
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
    # print(mid_proj)
    # mid2tar_projections.append(mid2tar[:, 0])
    # print(mid2tar[:1])
    

    # 상관관계 계산
    correlation = np.corrcoef(mid_proj[:, 0], tar_proj[:, 0])[0, 1]
    correlations.append((predict_interval, correlation))
    
    correlation = np.corrcoef(mid2tar[:, 0], tar_proj[:, 0])[0, 1]
    correlations_mid2tar.append((predict_interval, correlation))
    
    # correlation = np.corrcoef(mid_lat, tar_proj[:, 0])[0, 1]
    # correlations_mid_lat.append((predict_interval, correlation))

    correlation = np.corrcoef(mid_lat, tar_lat)[0, 1]
    correlations_mid_tar_lat.append((predict_interval, correlation))
    
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
    

    correlations_opt.append((predict_interval, correlation_opt))


    
# 상관관계 시각화
predict_intervals, corr_values = zip(*correlations)
print(corr_values)
plt.plot(predict_intervals, np.abs(corr_values), marker='o', label = 'Projection_each_spread')

# predict_intervals, corr_values = zip(*correlations_mid2tar)
# print(corr_values) 
# plt.plot(predict_intervals, np.abs(corr_values), marker='^', label = 'Projection_120h')

predict_intervals, corr_values = zip(*correlations_opt)
print(corr_values)
plt.plot(predict_intervals, np.abs(corr_values), marker='x', label = 'Projection_opt')

plt.legend(loc='lower right')
plt.xlabel('Predict Interval (hours)')
plt.ylabel('Correlation')
# plt.title(f'{key_str} - {target_str} Position correlation')
plt.ylim(0, 1)  
# plt.xlim(12, 120)  
plt.xlim(24, 120)  
plt.grid(True)

# x축 gridline 설정
plt.xticks(np.arange(24, 121, 24))
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)  # y=0 기준선 추가
plt.show()
# fig.savefig(f'{pangu_dir}/plot/Sensitivity/Position_correlation.png',bbox_inches='tight')


#%%
#! 가장 상관관계 높은 axis 구하기, 2nd main code
# 시간 지정
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,28,0)          #분석 시작 시점
target_time = datetime(2022,8,30,0)          #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")

total_time_range = int((target_time - key_time).total_seconds() / 3600)
start_time_range = int((start_time  - key_time).total_seconds() / 3600)

# 변수 지정
nearby_sign = 'n'                           #가까운 태풍만 추출
distance_threshold = 0.25                   #가까운 태풍의 거리
steering_sign = 'n'                         #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
steer_uni_alt = 0                           #steering wind를 구할 때, 고도를 하나
# choosen_factor_list = ['z','t','q']       #구하고자 하는 변수
choosen_factor_list = ['z']                 #구하고자 하는 변수
# altitude_list = [1000,850,700,500,300,200]  #각 변수에 대해 구하고자 하는 고도
altitude_list = [850,500,200]               #각 변수에 대해 구하고자 하는 고도
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
else:
    axis_opt = 'tar'  


# target_time 때도 살아있는 태풍만 추출
correlations = []
correlations_tar = []
correlations_opt = []
correlations_gg = []
correlations_df = []



for predict_interval in np.arange(start_time_range,total_time_range+1,6):
# for predict_interval in np.arange(54,total_time_range+1,6):
# for predict_interval in np.arange(start_time_range,48+1,6):
# for predict_interval in np.arange(36,37,6):
# for predict_interval in np.arange(120,124,6):
    fig_dir = f'/home1/jek/Pangu-Weather/plot/Sensitivity/{key_str}/{start_str}_{target_str}_{axis_opt}{nearby_sign_name}/'
    datetime1 = key_time + timedelta(hours=int(predict_interval))

    ens_num_list = []
    for ens in range(ens_num):
        if (target_time in ssv_dict[key_time][ens]) and (start_time in ssv_dict[key_time][ens]):
            ens_num_list.append(ens)
    
    print(len(ens_num_list))
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
    
    # 경도의 왜곡을 보정
    corr_pos_tar = np.copy(tar_pos)
    corr_pos_tar[:, 0] = (tar_pos[:, 0]-np.mean(tar_pos[:, 0])) * np.cos(np.radians(tar_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca_tar = PCA(n_components=1)
    pca_tar.fit(corr_pos_tar)
    pca_tar.mean_[0] = pca_tar.mean_[0] / np.cos(np.radians(pca_tar.mean_[1])) + np.mean(tar_pos[:, 0])

    
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
        ens_num_list = max_group
    
    
    

    corr_pos_mid = np.copy(mid_pos)
    corr_pos_mid[:, 0] = (mid_pos[:, 0]-np.mean(mid_pos[:, 0])) * np.cos(np.radians(mid_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca_mid = PCA(n_components=1)
    pca_mid.fit(corr_pos_mid)
    pca_mid.mean_[0] = pca_mid.mean_[0] / np.cos(np.radians(pca_mid.mean_[1])) + np.mean(mid_pos[:, 0])
    
    corr_pos_gg = np.copy(gg_pos)
    corr_pos_gg[:, 0] = (gg_pos[:, 0]-np.mean(gg_pos[:, 0])) * np.cos(np.radians(gg_pos[:, 1]))  # 경도에 cos(위도)를 곱해 거리 왜곡 보정
    pca_gg = PCA(n_components=1)
    pca_gg.fit(corr_pos_gg)
    pca_gg.mean_[0] = pca_gg.mean_[0] / np.cos(np.radians(pca_gg.mean_[1])) + np.mean(gg_pos[:, 0])
    
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

    
    
    mid_proj = pca_mid.transform(mid_pos)
    tar_proj = pca_tar.transform(tar_pos)
    gg_proj = pca_gg.transform(gg_pos)
    gg2tar = pca_tar.transform(gg_tar_pos)
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
                ens_factor_uv = np.load(f'{ens_factor_uv_path}/{predict_interval}h.npy')
                total_remove_uv = np.load(f'{total_remove_uv_path}/{predict_interval}h.npy')
                
                u_mean = np.load(f'{u_mean_path}/{predict_interval}h.npy')
                v_mean = np.load(f'{v_mean_path}/{predict_interval}h.npy')
                
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
                        ens_factor_uv = np.array(ens_factor_uv)[max_group_idx]
                    
                    if steering_sign == 'y':
                        u_mean = np.mean(ens_factor_uv[:,0,:,:], axis=0)
                        v_mean = np.mean(ens_factor_uv[:,1,:,:], axis=0)
            
            
            non_uv_mean = np.mean(ens_factor_uv, axis=0)
            
                # v_mean = v_mean[max_group_idx]
            # if axis_opt == 'quiver':
            #     ens_factor_z = []
            #     for ens in ens_num_list:
            #         output_data_dir = rf'{pangu_dir}/output_data/{first_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
            #         met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
                    
            #         ens_factor_z.append(met.met_data('z', level = 500))  
                        
            #     ens_factor_z = np.mean(np.array(ens_factor_z), axis=0)  
            #! deformation axis 구하기
            # tc_lon, tc_lat = tc_pos_mean

            # # 5도 이내 범위 찾기 => 추후 winfield solver에서 333km에 대해서만 하는 방안도 생각해보자
            # lon_min_index = np.where(lon_grid >= tc_lon - 3)[1][0]
            # lon_max_index = np.where(lon_grid <= tc_lon + 3)[1][-1]
            # lat_max_index = np.where(lat_grid >= tc_lat - 3)[0][-1]
            # lat_min_index = np.where(lat_grid <= tc_lat + 3)[0][0]

            # # 선택된 격자 포인트에 대한 데이터만 사용
            # selected_u = u_mean[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]
            # selected_v = v_mean[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]
            # selected_lon_grid = lon_grid[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]
            # selected_lat_grid = lat_grid[lat_min_index:lat_max_index+1, lon_min_index:lon_max_index+1]
            
        
            
            # # 수치 미분 계산
            # dx = 0.25  # 격자 간격(예시)
            # dy = 0.25  # 격자 간격(예시)

            # # 선택된 데이터에 대한 변형률 계산
            # du_dx = u_mean[lon_max_index] - u_mean[lon_min_index]
            # du_dy = u_mean[lat_max_index] - u_mean[lat_min_index]
            # dv_dx = v_mean[lon_max_index] - v_mean[lon_min_index]
            # dv_dy = v_mean[lat_max_index] - v_mean[lat_min_index]

            # shear = du_dy + dv_dx
            # stretch = du_dx - dv_dy
            # total_deformation = np.sqrt(shear**2 + stretch**2)
            
            # dilatation_axis_angle = 0.5 * np.arctan2(shear, stretch)
            # print(dilatation_axis_angle)
            # plt.quiver(selected_lon_grid, selected_lat_grid, np.cos(dilatation_axis_angle), np.sin(dilatation_axis_angle), cmap='coolwarm')
            # plt.colorbar()
            
            
            # 3도 떨어진 지점에서의 바람 데이터 추출
            # 각각 동서남북 방향으로 3도 떨어진 지점 인덱스 계산
            # indices = {
            #     'north': (np.argmin(np.abs(lat_grid[:, 0] - (tc_lat + 3))), np.argmin(np.abs(lon_grid[0] - tc_lon))),
            #     'south': (np.argmin(np.abs(lat_grid[:, 0] - (tc_lat - 3))), np.argmin(np.abs(lon_grid[0] - tc_lon))),
            #     'east':  (np.argmin(np.abs(lat_grid[:, 0] - tc_lat)), np.argmin(np.abs(lon_grid[0] - (tc_lon + 3)))),
            #     'west':  (np.argmin(np.abs(lat_grid[:, 0] - tc_lat)), np.argmin(np.abs(lon_grid[0] - (tc_lon - 3))))
            # }
            # u_north = u[indices['north']]
            # v_north = v[indices['north']]
            # u_south = u[indices['south']]
            # v_south = v[indices['south']]
            # u_east = u[indices['east']]
            # v_east = v[indices['east']]
            # u_west = u[indices['west']]
            # v_west = v[indices['west']]

            # # 변형률 계산
            # du_dx = (u_east - u_west) / (np.abs(lon_grid[0, indices['east'][1]] - lon_grid[0, indices['west'][1]]))
            # du_dy = (u_north - u_south) / (np.abs(lat_grid[indices['north'][0], 0] - lat_grid[indices['south'][0], 0]))
            # dv_dx = (v_east - v_west) / (np.abs(lon_grid[0, indices['east'][1]] - lon_grid[0, indices['west'][1]]))
            # dv_dy = (v_north - v_south) / (np.abs(lat_grid[indices['north'][0], 0] - lat_grid[indices['south'][0], 0]))

            # shear = du_dy + dv_dx
            # stretch = du_dx - dv_dy
            # total_deformation = np.sqrt(shear**2 + stretch**2)
            
            # rad_df = 0.5 * np.arctan2(shear, stretch)
            # df_direction = np.array([np.cos(rad_df), np.sin(rad_df)])
                     
            
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
            rad_gg  = np.arctan2(pca_gg.components_[0, 1] , pca_gg.components_[0, 0])
            
            # mid_df = mid_pos @ df_direction
            # correlation_df = np.corrcoef(mid_df, tar_proj[:, 0])[0, 1]
            # correlations_df.append((predict_interval, correlation_df))
            
            correlations_opt.append((predict_interval, best_correlation))
            
            correlation = np.corrcoef(mid_proj[:, 0], tar_proj[:, 0])[0, 1]
            correlations.append((predict_interval, correlation))
            
            correlation_gg = np.corrcoef(gg_proj[:, 0], gg2tar[:, 0])[0, 1]
            correlations_gg.append((predict_interval, correlation_gg))    
        
            
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
            
            pca_value = dir_corr[int(rad_gg /np.pi /2 * len(angles))]
            ax.quiver(rad_gg, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='purple', label='Group Direction')
            
            
            # pca_value = dir_corr[int(rad_df /np.pi /2 * len(angles))]
            # ax.quiver(rad_df, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='PCA Direction')
            ax.axvline(rad_tar, color='red', linewidth=2)
            
            ax.set_ylim(-1, 1)
            ax.set_theta_zero_location('E')  # 0도를 북쪽(위)으로 설정
            ax.set_yticks(np.arange(-1, 1.1, 0.5))

            # 저장 및 보여주기
            if not os.path.exists(f'{fig_dir}/Projection_direction/{start_str}_{target_str}{nearby_sign_name}'):
                os.makedirs(f'{fig_dir}/Projection_direction/{start_str}_{target_str}{nearby_sign_name}')
            else:
                pass
            plt.savefig(f'{fig_dir}/Projection_direction/{start_str}_{target_str}{nearby_sign_name}/{predict_interval}h.png')
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
                elif axis_opt == 'gg':
                    ens_factor = ens_factor[:,0,:,:]*pca_gg.components_[0, 0] + ens_factor[:,1,:,:]*pca_gg.components_[0, 1]
                elif axis_opt == 'lon':
                    ens_factor = ens_factor[:,0,:,:]
                elif axis_opt == 'lat':
                    ens_factor = ens_factor[:,1,:,:]
                elif axis_opt == 'quiver':
                    ens_factor_quiver = (ens_factor[:,0,:,:], ens_factor[:,1,:,:])
                
                
                
                if axis_opt == 'quiver':
                    cov_var_ratio = {}
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

            # 지도에 결과 표시
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            if axis_opt != 'quiver':
                contour = ax.contourf(lon_grid, lat_grid, cov_var_ratio, cmap=pwp, levels=np.linspace(-400, 400, 17), transform=ccrs.PlateCarree())
                cbar = plt.colorbar(contour, ax=ax, label=f'Cov(distance, {choosen_factor}) / Var({choosen_factor})', shrink = 0.6)
                cbar.locator = ticker.MultipleLocator(100)  # Set the colorbar ticks to have an interval of 0.5
                cbar.update_ticks()
            ax.scatter(mid_pos[:, 0], mid_pos[:, 1], alpha=0.7, s=0.6, c='green', zorder = 15)
            setup_map(ax, back_color='n')
            
           
            if steering_sign == 'y':
                # if axis_opt != 'quiver':
                ax.barbs(lon_grid[::8,::8], lat_grid[::8,::8], u_mean[::8,::8], v_mean[::8,::8], length=5, linewidth=0.5, color = 'black')

                        
                if axis_opt == 'opt':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], best_direction[0], best_direction[1], scale=20, color='b', width=0.003, label='Principal Axis')
                elif axis_opt == 'tar':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_tar.components_[0, 0], pca_tar.components_[0, 1], scale=20, color='r', width=0.003, label='Principal Axis')
                elif axis_opt == 'mid':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], pca_mid.components_[0, 0], pca_mid.components_[0, 1], scale=20, color='g', width=0.003, label='Principal Axis')
                elif axis_opt == 'gg':
                    ax.quiver(pca_gg.mean_[0], pca_gg.mean_[1], pca_gg.components_[0, 0], pca_gg.components_[0, 1], scale=20, color='purple', width=0.003, label='Principal Axis')
                elif axis_opt == 'lon':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 1, 0, scale=20, color='black', width=0.003, label='Principal Axis')
                elif axis_opt == 'lat':
                    ax.quiver(pca_mid.mean_[0], pca_mid.mean_[1], 0, 1, scale=20, color='black', width=0.003, label='Principal Axis')
                elif axis_opt == 'quiver':
                    # qui = ax.quiver(lon_grid[::8, ::8], lat_grid[::8, ::8], cov_var_ratio['u'][::8, ::8], cov_var_ratio['v'][::8, ::8], scale=10000, color='black', width=0.003)
                    # cax = ax.contourf(lon_grid, lat_grid, ens_factor_z, levels=np.arange(5520,6001,30), cmap='jet', zorder = 0, extend = 'both')
                    # fig.colorbar(cax, ax=ax, orientation='vertical', label='Geopotential Height (m)', shrink = 0.7)
                    qui = ax.quiver(lon_grid[::4, ::4], lat_grid[::4, ::4], cov_var_ratio['u'][::4, ::4], cov_var_ratio['v'][::4, ::4], scale=10000, color='blue', width=0.003)
                    # 먼저, quiverkey를 추가합니다.
                    key = ax.quiverkey(qui, X=0.9, Y=0.97, U=500, label='500 km', labelpos='E')

                    # 그런 다음, 배경을 추가합니다.
                    # quiverkey 추가

                    rect = patches.Rectangle((0.84, 0.94), 0.16, 0.06, linewidth=1.5, edgecolor='black', facecolor='white', transform=ax.transAxes)
                    ax.add_patch(rect)
                    
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
            
    #     if choosen_factor == 'u' or choosen_factor == 'v':
    #         total_remove_uv = np.swapaxes(total_remove_uv, 0, 1)
            
    #         for uv_uni in total_remove_uv:
    #             uv_uni_alt  = []
    #             uv_uni_1  = []
    #             uv_uni_2  = []

    #             for mp, uv, ens_uv in zip(mid_pos, uv_uni, ens_num_list):
    #                 # idx = np.where((lat_grid == mp[1]) & (lon_grid == mp[0])) # 오직 태풍 중심 위치 바람 정보 가져오기
    #                 #태풍 범위 내의 u 또는 v 평균 내기
    #                 dis_uv = haversine_distance(lat_grid, lon_grid, np.ones_like(lat_grid)*mp[1], np.ones_like(lat_grid)*mp[0])
    #                 # print(uv.shape)
    #                 # print(uv[dis_uv <= 333].shape)
                    
    #                 uv_uni_alt.append(np.mean(uv[dis_uv <= 333])) #태풍 중심으로부터 333km 이내의 모든 바람 데이터를 가져와서 평균
    #                 if ens_uv in group1:
    #                     uv_uni_1.append(np.mean(uv[dis_uv <= 333]))
    #                 if ens_uv in group2:
    #                     uv_uni_2.append(np.mean(uv[dis_uv <= 333]))
                    
    #             uv_uni_alt = np.mean(np.array(uv_uni_alt), axis = 0)
    #             uv_uni_1 = np.mean(np.array(uv_uni_1), axis = 0)
    #             uv_uni_2 = np.mean(np.array(uv_uni_2), axis = 0)
    #             uv_all_alt[choosen_factor].append(uv_uni_alt)
    #             uv_all_1[choosen_factor].append(uv_uni_1)
    #             uv_all_2[choosen_factor].append(uv_uni_2)
        
        
    #     uv_all_alt[choosen_factor] = np.array(uv_all_alt[choosen_factor])
    #     uv_all_1[choosen_factor] = np.array(uv_all_1[choosen_factor])
    #     uv_all_2[choosen_factor] = np.array(uv_all_2[choosen_factor])
           
            
    #     # print(uv[dis_uv <= 333].shape)   
    # if 'u' in choosen_factor_list and 'v' in choosen_factor_list:
        
    #     # 벡터 차이 계산
    #     delta_u = uv_all_2['u'] - uv_all_1['u']
    #     delta_v = uv_all_2['v'] - uv_all_1['v']

    #     # 벡터 차이를 이용한 방향 벡터 계산
    #     magnitude = np.sqrt(delta_u**2 + delta_v**2)
    #     gg_direction = np.array([delta_u, delta_v]) / magnitude

    #     print("Best direction to maximize the projection difference:", gg_direction)
        
        
    #     # plt.plot(uv_all_alt['u']*best_direction[0] + uv_all_alt['v']*best_direction[1], steer_pres,marker='o', label = 'opt_mean', color = 'orange')
    #     plt.plot(uv_all_1['u']*best_direction[0] + uv_all_1['v']*best_direction[1]-(uv_all_alt['u']*best_direction[0] + uv_all_alt['v']*best_direction[1]), steer_pres,marker = 'x', label = 'opt_group1', alpha = 0.5, color = 'orange')
    #     plt.plot(uv_all_2['u']*best_direction[0] + uv_all_2['v']*best_direction[1]-(uv_all_alt['u']*best_direction[0] + uv_all_alt['v']*best_direction[1]), steer_pres,marker = '^', label = 'opt_group2', alpha = 0.5, color = 'orange')
        
    #     # plt.plot(uv_all_alt['u']*pca_tar.components_[0, 0] + uv_all_alt['v']*pca_tar.components_[0, 1], steer_pres,marker='o', label = 'tar_mean', color = 'green')
    #     plt.plot(uv_all_1['u']*pca_tar.components_[0, 0] + uv_all_1['v']*pca_tar.components_[0, 1]-(uv_all_alt['u']*pca_tar.components_[0, 0] + uv_all_alt['v']*pca_tar.components_[0, 1]), steer_pres,marker = 'x', label = 'tar_group1', alpha = 0.5, color = 'green')
    #     plt.plot(uv_all_2['u']*pca_tar.components_[0, 0] + uv_all_2['v']*pca_tar.components_[0, 1]-(uv_all_alt['u']*pca_tar.components_[0, 0] + uv_all_alt['v']*pca_tar.components_[0, 1]), steer_pres,marker = '^', label = 'tar_group2', alpha = 0.5, color = 'green')
        
    #     # plt.plot(uv_all_alt['u'], steer_pres, marker='o', label = 'u_mean', color = 'blue')
    #     # plt.plot(uv_all_1['u'], steer_pres, marker='x', label = 'u_group1', alpha = 0.5, color = 'blue')
    #     # plt.plot(uv_all_2['u'], steer_pres, marker='^', label = 'u_group2', alpha = 0.5, color = 'blue')
    #     # plt.plot(uv_all_alt['v'], steer_pres, marker='o', label = 'v', color = 'red')
    #     # plt.plot(uv_all_1['v'], steer_pres, marker='x', label = 'v_group1', alpha = 0.5, color = 'red')
    #     # plt.plot(uv_all_2['v'], steer_pres, marker='^', label = 'v_group2', alpha = 0.5, color = 'red')
        
    #     plt.title(f'{predict_interval}h', fontweight = 'bold')
        
    #     plt.xlim(-1,1)
    #     plt.axvline(0, color = 'black')
    #     plt.gca().invert_yaxis()
    #     plt.legend()
    #     plt.grid()
    #     plt.xlabel('Wind speed($m/s$)')
    #     plt.ylabel('Pressure(hPa)')
    #     plt.show()       
        
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(111, polar=True)  # 극 좌표계 사용

    #     # 원형 그래프에 데이터 플로팅
    #     ax.plot(angles, dir_corr, label='Correlation', linewidth=2)  # 상관관계 선 그리기
    #     # 최대 상관관계 위치에 화살표 표시
    #     max_corr_radian = angles[max_index]
        
    #     # ax.quiver(max_corr_radian, 0, 0, best_correlation, angles='xy', scale_units='xy', scale=1, color='red', label='Max Correlation')
    #     ax.quiver(max_corr_radian, -1, 0, best_correlation + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='Max Corr')
    #     # PCA 방향에 화살표 표시
    #     pca_value = dir_corr[int(rad_pca /np.pi /2 * len(angles))]
    #     ax.quiver(rad_pca, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='green', label='PCA Direction')
    #     # pca_value = dir_corr[int(rad_df /np.pi /2 * len(angles))]
    #     # ax.quiver(rad_df, -1, 0, pca_value + 1, angles='xy', scale_units='xy', scale=1, color='blue', label='PCA Direction')
    #     ax.axvline(rad_tar, color='red', linewidth=2)
        
    #     ax.set_ylim(-1, 1)
    #     ax.set_theta_zero_location('E')  # 0도를 북쪽(위)으로 설정
    #     ax.set_yticks(np.arange(-1, 1.1, 0.5))

    #     # 저장 및 보여주기
    #     if not os.path.exists(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}'):
    #         os.makedirs(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}')
    #     else:
    #         pass
        # plt.savefig(f'{pangu_dir}/plot/Sensitivity/Projection_direction/{start_str}_{target_str}{nearby_sign_name}/{predict_interval}h.png')
        # plt.close()
        


# 상관관계 시각화
predict_intervals, corr_values = zip(*correlations)
plt.plot(predict_intervals, np.abs(corr_values), marker='o', label = 'Projected Each axis')
# predict_intervals, corr_values = zip(*correlations_tar)
# plt.plot(predict_intervals, np.abs(corr_values), marker='^', label = 'Projected final axis')
predict_intervals, corr_values = zip(*correlations_opt)
plt.plot(predict_intervals, np.abs(corr_values), marker='x', label = 'Projected Optimized axis')
# predict_intervals, corr_values = zip(*correlations_gg)
# plt.plot(predict_intervals, np.abs(corr_values), marker='x', label = 'Projected Group1,2 axis')
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
#! 이동속도 비교
# 시간 지정
key_time = datetime(2022,8,27,0)            #처음 시점 지정
start_time = datetime(2022,8,27,12)          #분석 시작 시점
target_time = datetime(2022,8,30,0)          #위치 projection을 구하고자 하는 시간

key_str = key_time.strftime("%m.%d %HUTC")
start_str = start_time.strftime("%m.%d %HUTC")
target_str = target_time.strftime("%m.%d %HUTC")

total_time_range = int((target_time - key_time).total_seconds() / 3600)
start_time_range = int((start_time  - key_time).total_seconds() / 3600)

# 변수 지정
nearby_sign = 'y'                           #가까운 태풍만 추출
distance_threshold = 0.5                    #가까운 태풍의 거리
steering_sign = 'y'                         #태풍 제거를 진행할 것인지를 판단(steering wind 진행)
steer_uni_alt = 0                           #steering wind를 구할 때, 고도를 하나
# choosen_factor_list = ['z','t','q']       #구하고자 하는 변수
choosen_factor_list = ['z']                 #구하고자 하는 변수
altitude_list = [1000,850,700,500,300,200]  #각 변수에 대해 구하고자 하는 고도
# altitude_list = [850,500,200]             #각 변수에 대해 구하고자 하는 고도
steer_pres = [850,700,600,500,400,300,250]  #steering wind 구할 때 사용하는 고도 바꿀 필요 x
axis_opt = 'quiver'                         #axis 뭘로 잡을지, opt: 위치 상관관계 최대인 axis, tar: 최종 위치의 axis, mid: 중간 위치의 axis, lon: 경도, lat: 위도
data_sign = 'y'                             #기존의 데이터를 사용할 것인지, n이면 새로 구함
save_sign = 'n'                             #!구한 데이터를 저장할 것인지, 만드는 코드 아니면 n으로 하자     


speed = {'mid':[], 'g1':[], 'g2':[]}
for predict_interval in np.arange(start_time_range+6,total_time_range+1,6):
# for predict_interval in np.arange(54,total_time_range+1,6):
# for predict_interval in np.arange(start_time_range,48+1,6):
# for predict_interval in np.arange(36,37,6):
# for predict_interval in np.arange(120,124,6):
    datetime1 = key_time + timedelta(hours=int(predict_interval))
    datetime0 = key_time + timedelta(hours=int(predict_interval)-6)

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
    mid_pos_p = [(ssv_dict[key_time][ens][datetime0]['lon'], ssv_dict[key_time][ens][datetime0]['lat']) for ens in ens_num_list]
    g1_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in group1]
    g1_pos_p = [(ssv_dict[key_time][ens][datetime0]['lon'], ssv_dict[key_time][ens][datetime0]['lat']) for ens in group1]
    g2_pos = [(ssv_dict[key_time][ens][datetime1]['lon'], ssv_dict[key_time][ens][datetime1]['lat']) for ens in group2]
    g2_pos_p = [(ssv_dict[key_time][ens][datetime0]['lon'], ssv_dict[key_time][ens][datetime0]['lat']) for ens in group2]
    
    mid_pos, mid_pos_p, g1_pos, g1_pos_p, g2_pos, g2_pos_p = np.array(mid_pos), np.array(mid_pos_p), np.array(g1_pos), np.array(g1_pos_p), np.array(g2_pos), np.array(g2_pos_p)
    
    mid_dis = list(map(haversine_distance, mid_pos[:,1], mid_pos[:,0], mid_pos_p[:,1], mid_pos_p[:,0]))
    g1_dis = list(map(haversine_distance, g1_pos[:,1], g1_pos[:,0], g1_pos_p[:,1], g1_pos_p[:,0]))
    g2_dis = list(map(haversine_distance, g2_pos[:,1], g2_pos[:,0], g2_pos_p[:,1], g2_pos_p[:,0]))
    
    mid_dis = mid_pos[:,0] - mid_pos_p[:,0]
    g1_dis = g1_pos[:,0] - g1_pos_p[:,0]
    g2_dis = g2_pos[:,0] - g2_pos_p[:,0]
    
    speed['mid'].append(np.mean(mid_dis))
    speed['g1'].append(np.mean(g1_dis))
    speed['g2'].append(np.mean(g2_dis))


plt.plot(speed['mid'], label = 'mid')
plt.plot(speed['g1'], label = 'g1') 
plt.plot(speed['g2'], label = 'g2')
plt.xticks(np.arange(0,10,1), np.arange(18,73,6))
plt.legend()