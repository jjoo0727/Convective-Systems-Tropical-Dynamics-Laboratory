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

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scipy.ndimage as ndimage
from scipy.stats import gaussian_kde
from scipy.ndimage import binary_dilation, minimum_filter, maximum_filter, label
from skimage.measure import regionprops

from datetime import datetime, timedelta

from haversine import haversine

import tropycal.tracks as tracks
import datetime as dt
import itertools  

def draw_and_connect_quarters(ax, center_lat, center_lon, distances, edgecolor='red'):
    path_coords = []
    for distance,(start_angle, end_angle) in zip(distances,[(0, 90), (90, 180), (180, 270), (270, 360)]):
        
        angles = np.linspace(start_angle, end_angle, 100)  # 더 많은 지점으로 호를 부드럽게
        
        for angle in angles:
            destination = geodesic(kilometers=distance).destination((center_lat, center_lon), angle)
            x, y = ccrs.PlateCarree().transform_point(destination.longitude, destination.latitude, ccrs.Geodetic())
            path_coords.append((x, y))
        
    path_coords.append((path_coords[0]))  # 시작점으로 돌아와 경로를 닫음
    path = Path(path_coords)
    patch = PathPatch(path,facecolor = 'none', edgecolor=edgecolor, linewidth=1, alpha=1)
    ax.add_patch(patch)
    
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

def storm_info(storm_name, storm_year, datetime_list=None):
    file_path = f'/Data/home/jjoo0727/Pangu-Weather/storm_info/{storm_year}_{storm_name}.csv'
    if not os.path.exists(file_path):
        ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc',catarina=True)
        storm = ibtracs.get_storm((storm_name,storm_year))
        storm = storm.to_dataframe()
        storm.to_csv(file_path, index=False)
    
    storm = pd.read_csv(file_path, parse_dates=['time'])
    first_index = np.where(storm['vmax']>=35)[0][0]
    
    
    # if datetime_list != None:
    first_time = datetime_list[datetime_list>=storm['time'][first_index:][0]][0]
    first_index = np.where(storm['time']>=first_time)[0][0]
    storm_lon = storm['lon'][first_index:]
    storm_lat = storm['lat'][first_index:]
    storm_mslp = storm['mslp'][first_index:]
    storm_time = storm['time'][first_index:]
    storm_time = np.array(storm_time.dt.to_pydatetime())
    return storm_lon.to_numpy(), storm_lat.to_numpy(), storm_mslp.to_numpy(), np.array(storm_time)

#%%
ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc',catarina=True)

#%%
# ibtracs.plot_summary(dt.datetime(2022,8,29,0),domain='west_pacific')
#https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/JTWC/awp122022.dat
storm = ibtracs.get_storm(('hinnamnor',2022))
forecasts = storm.get_operational_forecasts()
# ibtracs.get_storm_id(storm)


#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# forecast_dict['lat']

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax.coastlines(resolution='50m') 
# if back_color == 'y':
ocean_color = mcolors.to_rgba((147/255, 206/255, 229/255))
land_color = mcolors.to_rgba((191/255, 153/255, 107/255))
ax.add_feature(cfeature.OCEAN, color=ocean_color)
ax.add_feature(cfeature.LAND, color=land_color, edgecolor='none')

for i in range(1,31):
    if i < 10:
        i = '0'+str(i)
    forecast_dict = forecasts[f'AP{i}']['2022082806']
    cs = colorline(ax, forecast_dict['lon'], forecast_dict['lat'], z = forecast_dict['mslp'],norm=mcolors.Normalize(vmin=920, vmax=1020), linewidth=1)
    ax.set_extent([110,160,5,45], crs=ccrs.PlateCarree())


    # for lat, lon, windrad in zip(forecast_dict['lat'],forecast_dict['lon'], forecast_dict['windrad']):
    #     keys = list(windrad.keys())  # 키 리스트로 변환
    #     if 64 in keys:
    #         draw_and_connect_quarters(ax, lat, lon, windrad[64], edgecolor='red')
    #     if 50 in keys:
    #         draw_and_connect_quarters(ax, lat, lon, windrad[50], edgecolor='orange')
    #     if 34 in keys:
    #         draw_and_connect_quarters(ax, lat, lon, windrad[34], edgecolor='purple')


ax.coastlines()
plt.colorbar(cs, ax = ax, shrink = 0.8)
plt.show()

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# forecast_dict['lat']

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax.coastlines(resolution='50m') 
# if back_color == 'y':
ocean_color = mcolors.to_rgba((147/255, 206/255, 229/255))
land_color = mcolors.to_rgba((191/255, 153/255, 107/255))
ax.add_feature(cfeature.OCEAN, color=ocean_color)
ax.add_feature(cfeature.LAND, color=land_color, edgecolor='none')


forecast_dict = forecasts[f'JTWC']['2022083006']
cs = colorline(ax, forecast_dict['lon'], forecast_dict['lat'], z = forecast_dict['vmax'],norm=mcolors.Normalize(vmin=0, vmax=140), linewidth=1,cmap=plt.get_cmap('jet'))
ax.set_extent([110,160,5,45], crs=ccrs.PlateCarree())


for lat, lon, windrad in zip(forecast_dict['lat'],forecast_dict['lon'], forecast_dict['windrad']):
    keys = list(windrad.keys())  # 키 리스트로 변환
    if 64 in keys:
        draw_and_connect_quarters(ax, lat, lon, windrad[64], edgecolor='red')
    if 50 in keys:
        draw_and_connect_quarters(ax, lat, lon, windrad[50], edgecolor='orange')
    if 34 in keys:
        draw_and_connect_quarters(ax, lat, lon, windrad[34], edgecolor='purple')



ax.coastlines()
plt.colorbar(cs, ax = ax, shrink = 0.8)
plt.show()

#%%
# 
#태풍 지정
storm_name = 'hinnamnor'                                                                               
storm_name = storm_name.upper()
storm_year = 2022

#예측 시간 지정, 초기 시간 지정, 앙상블 수
predict_interval_list = np.arange(0,24*7+1,6)
first_str = '2022/08/27/00UTC'   
ens_num = 100

surface_factors = []  # 예시: 지표면에서는 'MSLP'만 선택
upper_factors = ['t'] 
perturation_scale = 0.2


surface_factors.sort()
upper_factors.sort()
surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가


first_time = datetime.strptime(first_str, "%Y/%m/%d/%HUTC")
final_time = first_time + timedelta(hours=int(predict_interval_list[-1]))
final_str = final_time.strftime("%Y/%m/%d/%HUTC")
datetime_list = np.array([first_time + timedelta(hours=int(hours)) for hours in predict_interval_list])

storm_lon, storm_lat, storm_mslp, storm_time = storm_info(storm_name, storm_year, datetime_list = datetime_list)   #태풍 영문명, 년도 입력

# nearby_storm = ibtracs.analogs_from_point((storm_lat[0],storm_lon[0]),radius=500, thresh={'v_max':35})

#%%

fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
# forecast_dict['lat']
space = 3

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='-')

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}
ax.coastlines(resolution='50m') 
# if back_color == 'y':
ocean_color = mcolors.to_rgba((147/255, 206/255, 229/255))
land_color = mcolors.to_rgba((191/255, 153/255, 107/255))
ax.add_feature(cfeature.OCEAN, color=ocean_color)
ax.add_feature(cfeature.LAND, color=land_color, edgecolor='none')
ax.set_extent([110,170,5,45])

for id, ibtr in ibtracs.data.items():
    if id.startswith('WP') and ibtr['name'] != 'UNNAMED':
        vmax_array = np.array(ibtr['vmax'])  # ibtr['vmax']를 numpy 배열로 변환합니다.
        valid_indices = np.where(vmax_array >= 35)[0]  # vmax가 35 이상인 모든 인덱스를 찾습니다.

        if len(valid_indices) == 0:  # vmax가 35 이상인 요소가 없으면 반복을 건너뜁니다.
            continue

        first_valid_index = valid_indices[0]  # 첫 번째 유효 인덱스를 가져옵니다.
        lon_start = np.array(ibtr['lon'])[first_valid_index]  # 해당 인덱스의 경도를 가져옵니다.
        lat_start = np.array(ibtr['lat'])[first_valid_index]  # 해당 인덱스의 위도를 가져옵니다.

        if storm_lon[0]-space <= lon_start <= storm_lon[0]+space and storm_lat[0]-space <= lat_start <= storm_lat[0]+space:
            cs = colorline(ax, ibtr['lon'], ibtr['lat'], z=ibtr['vmax'], norm=mcolors.Normalize(vmin=35, vmax=140), linewidth=1, cmap=plt.get_cmap('jet'))
            if np.max(vmax_array) > 80:
                print(ibtr['name'], ibtr['year'])
                
            lat_array = np.array(ibtr['lat'])
            if np.min(lat_array) < 10:
                print(ibtr['name'], ibtr['year'], 'strange')
cbar = plt.colorbar(cs, ax = ax, shrink=0.7)
cbar.set_label('kt', fontsize=15)
    # if id.startswith('WP') and ibtr['name'] != 'UNNAMED':
    #     # print(ibtr['name'])
    #     if ibtr['year'] == 1994:
    #         # print(ibtr['name'])
    #     # if ibtr['name'] == 'ELLIE':
    #         cs = colorline(ax, ibtr['lon'], ibtr['lat'], z=ibtr['vmax'], norm=mcolors.Normalize(vmin=920, vmax=1020), linewidth=1)
#%%
print(storm_lon[0])