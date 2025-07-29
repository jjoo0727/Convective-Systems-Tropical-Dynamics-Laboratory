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
from matplotlib.dates import DateFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation

from datetime import datetime, timedelta

import tropycal.tracks as tracks

import itertools    

from haversine import haversine


pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'

lat_indices = np.linspace(90, -90, 721)
lon_indices = np.linspace(-180, 180, 1441)[:-1]

pres_list = ['1000','925','850','700','600','500','400','300','250','200','150','100','50']
pres=500                                                #살펴볼 기압면 결정
p=pres_list.index(str(pres))

surface_factor = ['MSLP', 'U10', 'V10', 'T2M']
surface_dict = {'MSLP':0, 'U10':1, 'V10':2, 'T2M':3}
upper_factor = ['z', 'q', 't', 'u', 'v']
upper_dict = {'z':0, 'q':1, 't':2, 'u':3, 'v':4}

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

def storm_info(storm_name, storm_year):
    ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc',catarina=True)
    storm = ibtracs.get_storm((storm_name,storm_year))
    storm.to_dataframe()
    
    first_index = np.where((storm['vmax']>=35) == True)[0][0]
    storm_lon = storm['lon'][first_index:]-180
    storm_lat = storm['lat'][first_index:]
    storm_mslp = storm['mslp'][first_index:]
    storm_time = storm['time'][first_index:]
    return storm_lon, storm_lat, storm_mslp, storm_time

#태풍 발생 위치 주변 & 10m/s 이상 지역 주변
def plot_min_value(ax, data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position, mask_size=10, init_size=3):
    
    pred_str = datetime.strptime(pred_str, "%Y/%m/%d/%HUTC")
    init_str = storm_time[0]
    
    
    #태풍 발생 시점 이전엔 찾기 X
    if pred_str < init_str:
        return min_position
    
    storm_lon = storm_lon[storm_time<=pred_str]
    storm_lat = storm_lat[storm_time<=pred_str]
    storm_mslp = storm_mslp[storm_time<=pred_str]
    storm_time = storm_time[storm_time<=pred_str]
 
    # 데이터의 복사본 생성 및 마스킹
    data_copy = np.copy(data)
    wind_mask = wind_speed > 10         # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성
    expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((5, 5)))  # wind_mask의 주변 2픽셀 확장
    data_copy[~expanded_wind_mask] = np.nan # 확장된 마스크를 사용하여 wind_speed > 10 조건과 그 주변 2픽셀 이외의 위치를 NaN으로 설정
    
    if len(min_position) < 1:
        data_copy[(lat_grid > (storm_lat[0]+init_size))|(lat_grid < (storm_lat[0]-init_size))] = np.nan   
        data_copy[(lon_grid > (storm_lon[0]+init_size))|(lon_grid < (storm_lon[0]-init_size))] = np.nan 
    
    
    if pred_str > init_str and min_position:
        last_min_idx = min_position[-1][2]
        row_start = max(0, last_min_idx[0] - mask_size)
        row_end = min(data_copy.shape[0], last_min_idx[0] + mask_size + 1)  # +1은 Python의 슬라이싱이 상한을 포함하지 않기 때문
        col_start = max(0, last_min_idx[1] - mask_size)
        col_end = min(data_copy.shape[1], last_min_idx[1] + mask_size + 1)
        data_nan_filled = np.full(data_copy.shape, np.nan)
        data_nan_filled[row_start:row_end, col_start:col_end] = data_copy[row_start:row_end, col_start:col_end]
        data_copy = data_nan_filled

    if np.isnan(data_copy).all():
        print(pred_str, "모든 값이 NaN입니다. 유효한 최소값이 없습니다.")
    else:
        # 최소값 찾기
        min_index = np.unravel_index(np.nanargmin(data_copy), data_copy.shape)
        min_value = np.nanmin(data_copy)
        min_lat = lat_indices[lat_start + min_index[0]]
        min_lon = lon_indices[lon_start + min_index[1]]
        dis = haversine((min_lat, min_lon+180), (storm_lat[-1], storm_lon[-1]+180), unit = 'km')
        dis = (dis, pred_str)
        min_position.append([min_lon, min_lat, min_index, pred_str.strftime("%Y/%m/%d/%HUTC"), min_value, dis])
        # ax.text(min_lon, min_lat, f'{min_value/100:.0f}hPa', transform=ax.projection, color='gray', 
        #         horizontalalignment='center', verticalalignment='bottom', fontsize=20, fontweight='bold')
    
    norm_p = mcolors.Normalize(vmin=950, vmax=1020)
    
    for i, (lon, lat, idx, p_str, min_pres, dis) in enumerate(min_position):
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
    cbar_min_p = fig.colorbar(min_p, cax=min_ax, orientation='vertical')
    cbar_min_p.set_label('Minimum Pressure(hPa)', fontsize=16)
    ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=2)
    ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=2)
    ax.legend(loc='upper right')

    return min_position

#ax 배경 지정
def setup_map(ax):
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax.coastlines()
    ocean_color = mcolors.to_rgba((233/255, 240/255, 248/255))
    land_color = mcolors.to_rgba((255/255, 255/255, 255/255))
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
    z_850 = upper[upper_dict['z'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_500 = upper[upper_dict['z'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_200 = upper[upper_dict['z'], 9, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
    z_diff = z_200 - z_850
    
    # Calculate 10-meter wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    u_500 = upper[upper_dict['u'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v_500 = upper[upper_dict['v'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    u_850 = upper[upper_dict['u'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    v_850 = upper[upper_dict['v'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
    vorticity = calculate_vorticity(u_850, v_850, lat_grid, lon_grid)
    
    return mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500


#%%
#위경도 범위 지정
#동반구는 0~180, 서반구는 180~360y

proj = ccrs.PlateCarree(central_longitude=180)
original_cmap = plt.get_cmap("BrBG")
truncated_BrBG = truncate_colormap(original_cmap, minval=0.35, maxval=1.0) #수증기 colormap 지정

#위경도 지정
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])

#태풍 지정
storm_name = 'hinnamnor'                                                                               
storm_name = storm_name.upper()
storm_year = 2022
storm_lon, storm_lat, storm_mslp, storm_time = storm_info(storm_name, storm_year)   #태풍 영문명, 년도 입력

#예측 시간 지정
predict_interval_list = np.arange(0,24*7+1,6)[1:]     
year = ['2022']
month = ['08']
day = ['26','27','28']
times = ['00','12']

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


    predict_str = time_str
    surface = np.load(os.path.join(input_data_dir, 'surface.npy')).astype(np.float32)  
    upper = np.load(os.path.join(input_data_dir, 'upper.npy')).astype(np.float32)  
    mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500 = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{time_str} (+0h)\n', fontsize=20, loc = 'left')
    
    min_position = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                    wind_speed, predict_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position)
    setup_map(ax)
    weather_map_contour(ax, lon_grid, lat_grid, mslp)


    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{save_str}/vorticity'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{save_str}/vorticity')
    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_850'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_850')
    if not os.path.exists(f'{pangu_dir}/plot/typhoon/{save_str}/z_500'):
        os.makedirs(f'{pangu_dir}/plot/typhoon/{save_str}/z_500')
    # if not os.path.exists(f'{pangu_dir}/plot/typhoon/{time_str}/s_500'):
    #     os.makedirs(f'{pangu_dir}/plot/typhoon/{time_str}/s_500')
        
        
    contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4*5,10**-4*5,21),extend='both')
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Vorticity($s^{-1}$)', fontsize=16)  # You can customize the label here
    ax.set_title(f'850hPa Vorticity\n{storm_name}', fontsize=20, loc = 'right')
    if min_position:
        ax.set_title(f'850hPa Vorticity\n{storm_name} ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
    fig.savefig(f'{pangu_dir}/plot/typhoon/{save_str}/vorticity/0h.png', bbox_inches='tight')
    cbar.remove()  # colorbar 제거
    contourf.remove()
    
        

    contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet', levels = np.linspace(10000,11500,16))
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Thickness(m)', fontsize=16)  # You can customize the label here
    ax.set_title(f'200-850hPa Thickness\n{storm_name}', fontsize=20, loc = 'right')
    if min_position:
        ax.set_title(f'200-850hPa Thickness\n{storm_name} ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
    fig.savefig(f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_850/0h.png', bbox_inches='tight')
    cbar.remove()  # colorbar 제거
    contourf.remove()
    
        
    contourf = ax.contourf(lon_grid, lat_grid, z_500, cmap='jet', levels = np.arange(5220,6001,60))
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Height(m)', fontsize=16)  # You can customize the label here
    ax.set_title(f'500hPa Height\n{storm_name}', fontsize=20, loc = 'right')
    if min_position:
        ax.set_title(f'500hPa Height\n{storm_name} ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
    fig.savefig(f'{pangu_dir}/plot/typhoon/{save_str}/z_500/0h.png', bbox_inches='tight')
    cbar.remove()  # colorbar 제거
    contourf.remove()
        

    # strm = ax.streamplot(lon_grid, lat_grid, u_500, v_500, color=np.sqrt(u_500**2 + v_500**2), linewidth=1, density=4, cmap='jet')
    # cbar = fig.colorbar(strm.lines, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
    # cbar.set_label('Speed(m/s)')
    # ax.set_title(f'500hPa Streamline\n{storm_name}', fontsize=20, loc = 'right')
    # fig.savefig(f'{pangu_dir}/plot/typhoon/{time_str}/s_500/0h.png')
    
    plt.close(fig)

    #모델 예측 그리기
    for predict_interval in predict_interval_list:
        predict_time = time_obj + timedelta(hours=int(predict_interval))
        predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
        output_data_dir = rf'{pangu_dir}/output_data/{time_str}'
        # output_data_dir = rf'{pangu_dir}/output_data/{time_str[:4]}.{time_str[5:7]}.{time_str[8:10]} {time_str[11:]}'
        
        # Load surface data
        surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
        mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500 = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        
        fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
        ax.set_title(f'{time_str} (+{predict_interval}h)\n{predict_str}', fontsize=20, loc = 'left')
        min_position = plot_min_value(ax, mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid,
                                                        wind_speed, predict_str, storm_lon, storm_lat, storm_mslp, storm_time, min_position)
        setup_map(ax)
        weather_map_contour(ax, lon_grid, lat_grid, mslp)

        if min_position:
            if min_position[-1][5][1] == predict_time:
                dis_dic[time_obj][predict_time] = min_position[-1][5][0]
        
        contourf = ax.contourf(lon_grid, lat_grid, vorticity, cmap='seismic', levels = np.linspace(-10**-4*5,10**-4*5,21),extend='both')
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Vorticity($s^{-1}$)', fontsize=16)  # You can customize the label here
        ax.set_title(f'850hPa Vorticity\n{storm_name}', fontsize=20, loc = 'right')
        if min_position:
            ax.set_title(f'850hPa Vorticity\n{storm_name} ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
        fig.savefig(f'{pangu_dir}/plot/typhoon/{save_str}/vorticity/{predict_interval}h.png', bbox_inches='tight')
        cbar.remove()  # colorbar 제거
        contourf.remove()
        

        contourf = ax.contourf(lon_grid, lat_grid, z_diff, cmap='jet', levels = np.linspace(10000,11500,16))
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Thickness(m)', fontsize=16)  # You can customize the label here
        ax.set_title(f'200-850hPa Thickness\n{storm_name}', fontsize=20, loc = 'right')
        if min_position:
            ax.set_title(f'200-850hPa Thickness\n{storm_name} ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
        fig.savefig(f'{pangu_dir}/plot/typhoon/{save_str}/z_200-z_850/{predict_interval}h.png', bbox_inches='tight')
        cbar.remove()  # colorbar 제거
        contourf.remove()


        
        contourf = ax.contourf(lon_grid, lat_grid, z_500, cmap='jet', levels = np.arange(5220,6001,60))
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(contourf, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Height(m)', fontsize=16)  # You can customize the label here
        ax.set_title(f'500hPa Height\n{storm_name}', fontsize=20, loc = 'right')
        if min_position:
            ax.set_title(f'500hPa Height\n{storm_name} ({min_position[-1][4]/100:.0f}hPa)', fontsize=20, loc = 'right')
        fig.savefig(f'{pangu_dir}/plot/typhoon/{save_str}/z_500/{predict_interval}h.png', bbox_inches='tight')
        cbar.remove()  # colorbar 제거
        contourf.remove()


        # strm = ax.streamplot(lon_grid, lat_grid, u_500, v_500, color=np.sqrt(u_500**2 + v_500**2), linewidth=1, density=4, cmap='jet')
        # cbar = fig.colorbar(strm.lines, ax=ax, orientation='horizontal', extend='both', fraction=0.046, pad=0.04)
        # cbar.set_label('Speed(m/s)')
        # ax.set_title(f'500hPa Streamline\n{storm_name}', fontsize=20, loc = 'right')
        # fig.savefig(f'{pangu_dir}/plot/typhoon/{time_str}/s_500/{predict_interval}h.png')

        plt.close(fig)




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

# x축 레이블을 기울여서 날짜가 겹치지 않도록 함
# plt.xticks(rotation=45)

ax.set_ylabel("Distance (km)")

# 범례 표시
ax.legend(loc = 'upper left')
plt.tight_layout()
first_element = save_str.split('/')[0]
fig.savefig(f'{pangu_dir}/plot/typhoon/{first_element}/dis_error.png', bbox_inches='tight')
# 표시
plt.show()


#%%
min_position