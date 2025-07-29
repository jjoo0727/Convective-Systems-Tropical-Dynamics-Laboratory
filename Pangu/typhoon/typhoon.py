# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 01:03:11 2024

@author: jjoo0
"""
#%%
import os
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import plotly.figure_factory as ff
import matplotlib.collections as mcoll
from matplotlib.dates import DateFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation, minimum_filter, maximum_filter, label
from skimage.measure import regionprops

from datetime import datetime, timedelta

import tropycal.tracks as tracks

import itertools    

from haversine import haversine


pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'

# lat_indices = np.linspace(90, -90, 721)
# lon_indices = np.linspace(-180, 180, 1441)[:-1]
lat_indices = np.linspace(45, 5, 161)
lon_indices = np.linspace(-80, -20, 241)

pres_list = ['1000','925','850','700','600','500','400','300','250','200','150','100','50']
pres=500                                                #살펴볼 기압면 결정
p=pres_list.index(str(pres))

surface_factor = ['MSLP', 'U10', 'V10', 'T2M']
surface_dict = {'MSLP':0, 'U10':1, 'V10':2, 'T2M':3}
upper_factor = ['z', 'q', 't', 'u', 'v']
upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}

proj = ccrs.PlateCarree(central_longitude=180)
original_cmap = plt.get_cmap("BrBG")
# truncated_BrBG = truncate_colormap(original_cmap, minval=0.35, maxval=1.0) #수증기 colormap 지정
norm_p = mcolors.Normalize(vmin=950, vmax=1020)

#위경도 범위 지정 함수
def latlon_extent(lon_min, lon_max, lat_min, lat_max):    
    lon_min, lon_max = lon_min-180, lon_max-180  
     
    # 위경도 범위를 데이터의 행과 열 인덱스로 변환
    lat_start = np.argmin(np.abs(lat_indices - lat_max)) 
    lat_end = np.argmin(np.abs(lat_indices - lat_min))
    lon_start = np.argmin(np.abs(lon_indices - lon_min))
    lon_end = np.argmin(np.abs(lon_indices - lon_max))
    latlon_ratio = (lon_max-lon_min)/(lat_max-lat_min)
    extent=[lon_min, lon_max, lat_min, lat_max]
    return lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio

#수증기 색상 함수
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name, cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1.0):
    # x, y는 선의 좌표, z는 색상에 사용될 값
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    
    # z 값을 정규화
    z = np.asarray(z)

    # 선분을 색상으로 구분하여 그리기
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax.add_collection(lc)
    
    return lc

def calculate_vorticity(u, v, lat_grid, lon_grid):
    # Constants
    earth_radius = 6371e3  # in meters
    deg_to_rad = np.pi / 180

    # Pre-calculate deltas for longitude and latitude
    delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(lat_grid * deg_to_rad)
    delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(lat_grid)


    # Initialize vorticity array with nan values
    vorticity = np.full_like(u, np.nan)

    # Calculate partial derivatives using slicing, avoiding boundaries
    dv_dx = np.empty_like(v)
    dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * delta_lon[:, 1:-1])
    dv_dx[:, 0] = dv_dx[:, -1] = np.nan

    du_dy = np.empty_like(u)
    du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * delta_lat[1:-1, :])
    du_dy[0, :] = du_dy[-1, :] = np.nan

    # Calculate vorticity avoiding boundaries
    vorticity[1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]

    return vorticity

def storm_info(storm_name, storm_year, datetime_list=None):
    ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc',catarina=True)
    storm = ibtracs.get_storm((storm_name,storm_year))
    storm.to_dataframe()
    first_index = np.where(storm['vmax']>=35)[0][0]
    
    if datetime_list != None:
        datetime_list = np.array(datetime_list)
        first_time = datetime_list[datetime_list>=storm['time'][first_index:][0]][0]
        first_index = np.where(storm['time']>=first_time)[0][0]
    # print(first_index)
    storm_lon = storm['lon'][first_index:]-180
    storm_lat = storm['lat'][first_index:]
    storm_mslp = storm['mslp'][first_index:]
    storm_time = storm['time'][first_index:]
    return storm_lon, storm_lat, storm_mslp, storm_time

#태풍 발생 위치 주변 & 10m/s 이상 지역 주변
def plot_min_value(data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_str, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                   min_position, mask_size=10, init_size=10, cb_num=0, ax = None):
    
    pred_str = datetime.strptime(pred_str, "%Y/%m/%d/%HUTC")
    init_str = storm_time[0]
    
    
    storm_lon = storm_lon[storm_time<=pred_str]
    storm_lat = storm_lat[storm_time<=pred_str]
    storm_mslp = storm_mslp[storm_time<=pred_str]
    storm_time = storm_time[storm_time<=pred_str]
 
    # 데이터의 복사본 생성 및 마스킹
    data_copy = np.copy(data)
    
    filtered_data = minimum_filter(data_copy, size=21)
    local_minima = data_copy == filtered_data

    # 로컬 최소값의 위치 찾기
    minima_labels, num_features = label(local_minima)
    minima_positions = np.array([np.mean(np.where(minima_labels == i), axis=1) for i in range(1, num_features+1)])

    z_minima = z_diff == maximum_filter(z_diff, size=21)
    z_labels, z_num_features = label(z_minima)
    z_positions = np.array([np.mean(np.where(z_labels == i), axis=1) for i in range(1, z_num_features+1)])
    
    # lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
    # lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
    
    # x_pos = np.array([int(i[0]) for i in minima_positions])
    # y_pos = np.array([int(i[1]) for i in minima_positions])
    # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    # setup_map(ax,back_color='n')
    # ax.set_extent(extent, crs=proj)
    # ax.set_title(f'{pred_str} MSLP', fontsize=20)
    # mslp_ax = ax.scatter(lon_grid, lat_grid, c=data/100)
    # plt.colorbar(mslp_ax, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
    # ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
    # plt.show()
    
    # x_pos = np.array([int(i[0]) for i in minima_positions])
    # y_pos = np.array([int(i[1]) for i in minima_positions])
    # fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    # setup_map(ax,back_color='n')
    # ax.set_extent(extent, crs=proj)
    # ax.set_title(f'{pred_str} Z_diff', fontsize=20)
    # z_ax = ax.scatter(lon_grid, lat_grid, c=z_diff)
    # plt.colorbar(z_ax, ax=ax, orientation='vertical', label='Z_diff (m)', shrink=0.8)
    # ax.scatter(lon_indices[lon_start +y_pos], lat_indices[lat_start + x_pos], s=20, c='red')
    # print(lon_indices[lon_start +y_pos]+180, lat_indices[lat_start + x_pos])
    # plt.show()
    
    #태풍 발생 시점 이전엔 찾기 X
    if pred_str < init_str:
        return min_position
    
    fm_positions = []

    # 각 z_position에 대해 모든 minima_positions까지의 거리를 계산
    for z_pos in z_positions:
        for min_pos in minima_positions:
            # 유클리드 거리 계산
            distance = np.sqrt((z_pos[0] - min_pos[0])**2 + (z_pos[1] - min_pos[1])**2)
            
            # 거리가 5 이하인 경우만 선택
            if distance <= 10:
                fm_positions.append(min_pos)

    # 중복 위치 제거
    minima_positions = np.unique(fm_positions, axis=0)

    wind_mask = wind_speed > 8         # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성

    expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((15,15)))  # wind_mask의 주변 2픽셀 확장
    data_copy[~expanded_wind_mask] = np.nan # 확장된 마스크를 사용하여 wind_speed > 10 조건과 그 주변 2픽셀 이외의 위치를 NaN으로 설정
    
    if len(min_position) < 1:
        data_copy[(lat_grid > (storm_lat[0]+init_size))|(lat_grid < (storm_lat[0]-init_size))] = np.nan   
        data_copy[(lon_grid > (storm_lon[0]+init_size))|(lon_grid < (storm_lon[0]-init_size))] = np.nan 
    

    if pred_str > init_str and min_position:
        last_min_idx = min_position[-1][2]
        # if len(min_position)>1:
        #     lat_pred = min_position[-1][1]+(min_position[-1][1]-min_position[-2][1])
        #     if lat_pred > 30:
        #         mask_size=int(mask_size*1.2)
        #     elif lat_pred > 35:
        #         mask_size=int(mask_size*2)
        
        # 로컬 최소값 찾기
        
        
        row_start = max(0, last_min_idx[0] - mask_size)
        row_end = min(data_copy.shape[0], last_min_idx[0] + mask_size + 1)  # +1은 Python의 슬라이싱이 상한을 포함하지 않기 때문
        col_start = max(0, last_min_idx[1] - mask_size)
        col_end = min(data_copy.shape[1], last_min_idx[1] + mask_size + 1)
        data_nan_filled = np.full(data_copy.shape, np.nan)
        data_nan_filled[row_start:row_end, col_start:col_end] = data_copy[row_start:row_end, col_start:col_end]
        data_copy = data_nan_filled
        # plt.imshow(data_copy)
        # plt.show()


    if np.isnan(data_copy).all():
        print(pred_str, "모든 값이 NaN입니다. 유효한 최소값이 없습니다.")

    else:
        # 최소값 찾기
        filtered_positions = []
        for pos in minima_positions:
            lat, lon = int(pos[1]), int(pos[0])
            if not np.isnan(data_copy[lon, lat]):
                filtered_positions.append((int(lon), int(lat)))

        minima_positions = np.array(filtered_positions)
        # print(minima_positions)
        # print(minima_positions)
        

    

        
        if (len(minima_positions) < 1) and (len(min_position)>0):  #태풍 소멸 이후
            if min_position[-1][6] == 'tc':
                print(pred_str, "태풍이 소멸하였습니다.")
                min_position[-1][6] = 'ex'
            
        elif (len(minima_positions) < 1) and (len(min_position)<1): #태풍 발생을 못 찾음
            pass
        
        elif (len(minima_positions) > 0) and (len(min_position)>0):
            dis_pos_list=[]

            if min_position[-1][6] == 'tc':
                for pos in minima_positions:
                    # print('pos', pos)
                    min_index = (pos[0], pos[1])
                    min_lat = lat_indices[lat_start + pos[0]]
                    min_lon = lon_indices[lon_start + pos[1]]
                    min_value = data_copy[pos[0],pos[1]]
                    error = haversine((min_lat, min_lon+180), (storm_lat[-1], storm_lon[-1]+180), unit = 'km')
                    error = (error, pred_str)
                    dis_pos = haversine((min_lat, min_lon+180), (min_position[-1][1], min_position[-1][0]+180), unit = 'km')
                    dis_pos_list.append([min_lat, min_lon, min_index, min_value, error, dis_pos])
                    if len(dis_pos_list)>1:
                        if dis_pos_list[-1][5] > dis_pos_list[-2][5]:
                            dis_pos_list.pop()
                
                dis_pos_list = dis_pos_list[0]
                min_lat, min_lon, min_index, min_value, error = dis_pos_list[0], dis_pos_list[1], dis_pos_list[2], dis_pos_list[3], dis_pos_list[4]
                min_position.append([min_lon, min_lat, min_index, pred_str.strftime("%Y/%m/%d/%HUTC"), min_value, error,'tc'])
            
            else:
                pass
                
        elif (len(minima_positions) > 0) and (len(min_position)<1):
            min_index = minima_positions[0]
            min_value = data_copy[int(minima_positions[0][0]),int(minima_positions[0][1])]
            min_lat = lat_indices[lat_start + min_index[0]]
            min_lon = lon_indices[lon_start + min_index[1]]
            error = haversine((min_lat, min_lon+180), (storm_lat[-1], storm_lon[-1]+180), unit = 'km')
            error = (error, pred_str)
            
            min_position.append([min_lon, min_lat, min_index, pred_str.strftime("%Y/%m/%d/%HUTC"), min_value, error,'tc'])
                # print('min_indx', min_position[-1][2])
            # ax.text(min_lon, min_lat, f'{min_value/100:.0f}hPa', transform=ax.projection, color='gray', 
            #         horizontalalignment='center', verticalalignment='bottom', fontsize=20, fontweight='bold')
    
    if ax != None:
        norm_p = mcolors.Normalize(vmin=950, vmax=1020)
        
        for i, (lon, lat, idx, p_str, min_pres, dis, _) in enumerate(min_position):
            if p_str.endswith('00UTC'):
                dx, dy = 3, -3  # 시간 나타낼 위치 조정
                new_lon, new_lat = lon + dx, lat + dy
                
                # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
                ax.annotate(p_str[5:], xy=(lon, lat), xytext=(new_lon, new_lat),
                        textcoords='data', arrowprops=dict(arrowstyle="-", color='gray'),
                        color='gray', horizontalalignment='center', verticalalignment='center', fontsize=8,
                        transform=ax.projection)


        lons = [pos[0] for pos in min_position]
        lats = [pos[1] for pos in min_position]
        min_values = [pos[4]/100 for pos in min_position]
        
        if len(min_position) > 1:
            ax.plot(lons, lats, color='red', transform=ax.projection, linestyle='-', marker='', label = 'model pred', zorder=3)
        
        min_p = ax.scatter(lons, lats, c=min_values, cmap='jet_r', norm=norm_p, transform=ax.projection, zorder=3)
        min_ax = fig.add_axes([1, 0.15, 0.03, 0.7])
        if cb_num == 0:
            cbar_min_p = fig.colorbar(min_p, cax=min_ax, orientation='vertical')
            cbar_min_p.set_label('Minimum Pressure(hPa)', fontsize=16)
        ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=2)
        ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=2)
        ax.legend(loc='upper right')

    return min_position


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
    

def process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid):
    
    # Extract surface variables
    mslp = surface[surface_dict['MSLP']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    u10 = surface[surface_dict['U10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v10 = surface[surface_dict['V10']][lat_start:lat_end + 1, lon_start:lon_end + 1]
    
    # Calculate geopotential heights and their difference
    # z_850 = upper[upper_dict['z'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_500 = upper[upper_dict['z'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_200 = upper[upper_dict['z'], 9, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_diff = z_200 - z_500
    
    # Calculate 10-meter wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    u_500 = upper[upper_dict['u'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v_500 = upper[upper_dict['v'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    u_850 = upper[upper_dict['u'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v_850 = upper[upper_dict['v'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    vorticity = calculate_vorticity(u_850, v_850, lat_grid, lon_grid)
    
    return mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500


def contourf_and_save(ax, fig, lon_grid, lat_grid, data, min_position, 
                      title='', label='', levels=None, cmap='jet', save_path=''):
    
    contourf = ax.contourf(lon_grid, lat_grid, data, cmap=cmap, levels=levels, extend='both')
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
    cbar.set_label(label, fontsize=16)
    ax.set_title(title, fontsize=20, loc = 'right')
    if min_position:
        ax.set_title(title+f' ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
    plt.savefig(save_path, bbox_inches='tight')
    cbar.remove()
    contourf.remove()  # 이 방법으로 contourf 객체를 제거
    
#%%
#여러 시점 비교
#위경도 범위 지정
#동반구는 0~180, 서반구는 180~360y

#위경도 지정
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

#태풍 지정
storm_name = 'maysak'                                                                               
storm_name = storm_name.upper()
storm_year = 2020
# storm_lon, storm_lat, storm_mslp, storm_time = storm_info(storm_name, storm_year)   #태풍 영문명, 년도 입력

#예측 시간 지정
predict_interval_list = np.arange(0,24*7+1,6)     
year = ['2020']
month = ['08']
day = ['26','27','28']
times = ['00','06','12','18']

dis_dic = {}

for y, m, d, tm in itertools.product(year, month, day, times):
    #ERA5 initial map
    time_str = f'{y}/{m}/{d}/{tm}UTC'
    save_str = f'{y}_{storm_name}/{m}/{d}/{tm}UTC'
    time_obj = datetime.strptime(time_str, "%Y/%m/%d/%HUTC")
    print(time_str)
    input_data_dir = rf'{pangu_dir}/input_data/{time_str}'
    
    min_position = []  # 최저값 위치를 저장할 리스트
    dis_dic[time_obj] = {}



    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{save_str}/vorticity'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{save_str}/vorticity')
    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_500'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_500')
    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{save_str}/z_500'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{save_str}/z_500')
        
    for predict_interval in predict_interval_list:
    #모델 예측 그리기
    
        predict_time = time_obj + timedelta(hours=int(predict_interval))
        predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
        output_data_dir = rf'{pangu_dir}/output_data/{time_str}/1ENS'
        # output_data_dir = rf'{pangu_dir}/output_data/{time_str[:4]}.{time_str[5:7]}.{time_str[8:10]} {time_str[11:]}'
        
        # Load surface data
        surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
        mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500 = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        
        fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        ax.set_title(f'{time_str} (+{predict_interval}h)\n{predict_str}', fontsize=20, loc = 'left')
        min_position = plot_min_value(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                        wind_speed, predict_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position, ax=ax)
        setup_map(ax)
        weather_map_contour(ax, lon_grid, lat_grid, mslp)

        if min_position:
            if min_position[-1][5][1] == predict_time:
                dis_dic[time_obj][predict_time] = min_position[-1][5][0]
        
        
        contourf_and_save(ax, fig, lon_grid, lat_grid, vorticity, min_position, 
                      title=f'850hPa Vorticity\n{storm_name}', label='Vorticity($s^{-1}$)', 
                      levels=np.linspace(-10**-4*5,10**-4*5,21), cmap='seismic', save_path=f'{pangu_dir}/plot/typhoon/{save_str}/vorticity/{predict_interval}h.png')
    
        contourf_and_save(ax, fig, lon_grid, lat_grid, z_diff, min_position, 
                        title=f'200-500hPa Thickness\n{storm_name}', label='Thickness(m)', 
                        levels=np.linspace(5000,7000,21), cmap='jet', save_path=f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_500/{predict_interval}h.png')
        
        contourf_and_save(ax, fig, lon_grid, lat_grid, z_500, min_position, 
                        title=f'500hPa Height\n{storm_name}', label='Height(m)', 
                        levels=np.arange(5220,6001,60), cmap='jet', save_path=f'{pangu_dir}/plot/typhoon/{save_str}/z_500/{predict_interval}h.png')


        # strm = ax.streamplot(lon_grid, lat_grid, u_500, v_500, color=np.sqrt(u_500**2 + v_500**2), linewidth=1, density=4, cmap='jet')
        # cbar = fig.colorbar(strm.lines, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
        # cbar.set_label('Speed(m/s)')
        # ax.set_title(f'500hPa Streamline\n{storm_name}', fontsize=20, loc = 'right')
        # fig.savefig(f'{pangu_dir}/plot/typhoon/{time_str}/s_500/{predict_interval}h.png')

        plt.close(fig)



#모델 오차 그리기
fig, ax = plt.subplots()
ax.set_title(f'{storm_name} Prediction Distance Error')

for time_obj, predictions in dis_dic.items():
    if time_obj.hour == 0 or time_obj.hour == 12:
        predict_times = sorted(predictions.keys())
        values = [predictions[p_time] for p_time in predict_times]
        
        # 선 그래프 그리기
        ax.plot(predict_times, values, label=time_obj.strftime('%m-%d %HUTC'))

# 날짜 포맷터 설정
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)

ax.set_ylabel("Distance (km)")

# 범례 표시
ax.legend(loc = 'upper left')
plt.tight_layout()
first_element = save_str.split('/')[0]
fig.savefig(f'{pangu_dir}/plot/typhoon/{first_element}/dis_error.png', bbox_inches='tight')
# 표시
plt.show()
