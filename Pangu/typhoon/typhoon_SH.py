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


#와도 계산 함수
def calculate_vorticity(u, v, lat_grid, level = None):
    # Constants
    earth_radius = 6371e3  # in meters
    deg_to_rad = np.pi / 180

    # Pre-calculate deltas for longitude and latitude
    delta_lon = 0.25 * deg_to_rad * earth_radius * np.cos(lat_grid * deg_to_rad)
    delta_lat = 0.25 * deg_to_rad * earth_radius * np.ones_like(lat_grid)


    # Initialize vorticity array with nan values
    vorticity = np.full_like(u, np.nan)

    # Calculate partial derivatives using slicing, avoiding boundaries
    if level == None:
        dv_dx = np.empty_like(v)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * delta_lon[:, 1:-1])
        dv_dx[:, 0] = dv_dx[:, -1] = np.nan

        du_dy = np.empty_like(u)
        du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * delta_lat[1:-1, :])
        du_dy[0, :] = du_dy[-1, :] = np.nan

        # Calculate vorticity avoiding boundaries
        vorticity[1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]
    
    else:
        for i in range(13):
            dv_dx = np.empty_like(v[i])
            dv_dx[:, 1:-1] = (v[i, :, 2:] - v[i, :, :-2]) / (2 * delta_lon[:, 1:-1])
            dv_dx[:, 0] = dv_dx[:, -1] = np.nan

            du_dy = np.empty_like(u[i])
            du_dy[1:-1, :] = (u[i, :-2, :] - u[i, 2:, :]) / (2 * delta_lat[1:-1, :])
            du_dy[0, :] = du_dy[-1, :] = np.nan

            # Calculate vorticity avoiding boundaries
            vorticity[i,1:-1, 1:-1] = dv_dx[1:-1, 1:-1] - du_dy[1:-1, 1:-1]

    return vorticity


#태풍 정보 불러오는 함수
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


#output 기상 정보 클래스
class Met:
    def __init__(self, output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid):
        self.surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        self.upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
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
        
        return calculate_vorticity(u, v, self.lat_grid, level)

#태풍 발생 위치 주변 & 10m/s 이상 지역 주변
def plot_min_value(data, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
                   wind_speed, pred_str, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
                   min_position, mask_size=10, init_size=10, cb_num=0, ax = None):
    
    pred_str = datetime.strptime(pred_str, "%Y/%m/%d/%HUTC")
    init_str = storm_time[0]
    
    
    #pred_str 전의 시간의 태풍 정보만 불러옴
    storm_lon = storm_lon[storm_time<=pred_str]
    storm_lat = storm_lat[storm_time<=pred_str]
    storm_mslp = storm_mslp[storm_time<=pred_str]
    storm_time = storm_time[storm_time<=pred_str]
 
    # 로컬 최소값의 위치 찾기
    data_copy = np.copy(data)   #data_copy는 MSLP 정보
    filtered_data = minimum_filter(data_copy, size=21)
    local_minima = data_copy == filtered_data
    minima_labels, num_features = label(local_minima)
    minima_positions = np.array([np.mean(np.where(minima_labels == i), axis=1) for i in range(1, num_features+1)])

    z_minima = z_diff == maximum_filter(z_diff, size=21)
    z_labels, z_num_features = label(z_minima)
    z_positions = np.array([np.mean(np.where(z_labels == i), axis=1) for i in range(1, z_num_features+1)])
    
    '''mslp, z 그리기
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
    # plt.show()'''
    
    #태풍 발생 시점 이전엔 찾기 X
    if pred_str < init_str:
        return min_position
    
    fm_positions = []

    # 각 z_position에 대해 모든 minima_positions까지의 거리를 계산 후 일정 거리 이내의 minima_position만 취득
    for z_pos in z_positions:
        for min_pos in minima_positions:
            # 유클리드 거리 계산
            distance = np.sqrt((z_pos[0] - min_pos[0])**2 + (z_pos[1] - min_pos[1])**2)
            
            # 거리가 5 이하인 경우만 선택
            if distance <= 10:
                fm_positions.append(min_pos)

    
    minima_positions = np.unique(fm_positions, axis=0) # 중복 위치 제거

    wind_mask = wind_speed > 8         # wind_speed > 10인 조건을 만족하는 픽셀에 대한 마스크 생성

    expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((15,15)))  # wind_mask의 주변 2픽셀 확장
    data_copy[~expanded_wind_mask] = np.nan # 확장된 마스크를 사용하여 wind_speed > 10 조건과 그 주변 2픽셀 이외의 위치를 NaN으로 설정
    
    if len(min_position) < 1:
        data_copy[(lat_grid > (storm_lat[0]+init_size))|(lat_grid < (storm_lat[0]-init_size))] = np.nan   
        data_copy[(lon_grid > (storm_lon[0]+init_size))|(lon_grid < (storm_lon[0]-init_size))] = np.nan 
    

    if pred_str > init_str and min_position:
        last_min_idx = min_position[-1]['idx']
        
        
        row_start = max(0, last_min_idx[0] - mask_size)
        row_end = min(data_copy.shape[0], last_min_idx[0] + mask_size + 1)  # +1은 Python의 슬라이싱이 상한을 포함하지 않기 때문
        col_start = max(0, last_min_idx[1] - mask_size)
        col_end = min(data_copy.shape[1], last_min_idx[1] + mask_size + 1)
        data_nan_filled = np.full(data_copy.shape, np.nan)
        data_nan_filled[row_start:row_end, col_start:col_end] = data_copy[row_start:row_end, col_start:col_end]
        data_copy = data_nan_filled
        # plt.imshow(data_copy)
        # plt.show()

    #data_copy의 모든 값들이 Nan이면 패스
    if np.isnan(data_copy).all():
        print(pred_str, "모든 값이 NaN입니다. 유효한 최소값이 없습니다.")

    #data_copy에서 최소값 찾기
    else:
        #data_copy에서 nan이 아닌 부분에서만 minima_position 살리기
        filtered_positions = []
        for pos in minima_positions:
            lat, lon = int(pos[1]), int(pos[0])
            if not np.isnan(data_copy[lon, lat]):
                filtered_positions.append((int(lon), int(lat)))

        minima_positions = np.array(filtered_positions)

        
        if (len(minima_positions) < 1) and (len(min_position)>0):  #태풍 소멸 이후
            if min_position[-1]['type'] == 'tc':
                print(pred_str, "태풍이 소멸하였습니다.")
                min_position[-1]['type'] = 'ex'
            
        elif (len(minima_positions) < 1) and (len(min_position)<1): #태풍 발생을 못 찾음
            pass
        
        elif (len(minima_positions) > 0) and (len(min_position)>0):
            dis_pos_list=[]

            if min_position[-1]['type'] == 'tc':
                for pos in minima_positions:
                    # print('pos', pos)
                    min_index = (pos[0], pos[1])
                    min_lat = lat_indices[lat_start + pos[0]]
                    min_lon = lon_indices[lon_start + pos[1]]
                    min_value = data_copy[pos[0],pos[1]]
                    error = haversine((min_lat, min_lon+180), (storm_lat[-1], storm_lon[-1]+180), unit = 'km')
                    error = (error, pred_str)
                    # 여러 minima가 있는 경우 우열을 가림. 이전보다 더 먼 곳에 위치한 minima가 pop됨
                    dis_pos = haversine((min_lat, min_lon+180), (min_position[-1]['lat'], min_position[-1]['lon']+180), unit = 'km')
                    dis_pos_list.append([min_lat, min_lon, min_index, min_value, error, dis_pos])
                    if len(dis_pos_list)>1:
                        if dis_pos_list[-1][5] > dis_pos_list[-2][5]:
                            dis_pos_list.pop()
                
                min_lat, min_lon, min_index, min_value, error = dis_pos_list[0][0], dis_pos_list[0][1], dis_pos_list[0][2], dis_pos_list[0][3], dis_pos_list[0][4]
                min_position.append({'lon': min_lon, 'lat': min_lat, 'idx': min_index, 
                                     'time': pred_str.strftime("%Y/%m/%d/%HUTC"), 'mslp':  min_value, 'error': error, 'type':'tc'})
            
            else:
                pass
                
        elif (len(minima_positions) > 0) and (len(min_position)<1):
            min_index = minima_positions[0]
            min_value = data_copy[int(minima_positions[0][0]),int(minima_positions[0][1])]
            min_lat = lat_indices[lat_start + min_index[0]]
            min_lon = lon_indices[lon_start + min_index[1]]
            error = haversine((min_lat, min_lon+180), (storm_lat[-1], storm_lon[-1]+180), unit = 'km')
            error = (error, pred_str)
            
            min_position.append({'lon': min_lon, 'lat': min_lat, 'idx': min_index, 'time': pred_str.strftime("%Y/%m/%d/%HUTC"), 'mslp':  min_value, 'error': error, 'type':'tc'})
    
    
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


        lons = [pos['lon'] for pos in min_position]
        lats = [pos['lat'] for pos in min_position]
        min_values = [pos['mslp'] for pos in min_position]
        
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


    

#수증기 색상 함수
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'truncated_' + cmap.name, cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#점이 아닌 선의 색상으로 강도를 나타내는 함수
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
        ax.set_title(title+f' ({min_position[-1]["mslp"]:.0f}hPa)', fontsize=20, loc = 'right')
    plt.savefig(save_path, bbox_inches='tight')
    cbar.remove()
    contourf.remove()  # 이 방법으로 contourf 객체를 제거



# %%
#위경도 지정
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(100,160,5,45)  
lon_grid, lat_grid = np.meshgrid(lon_indices[lon_start:lon_end + 1], lat_indices[lat_start:lat_end + 1])
# lon_grid -= 180


#태풍 지정
storm_name = 'hinnamnor'                                                                               
storm_name = storm_name.upper()
storm_year = 2022


#예측 시간 지정, 초기 시간 지정, 앙상블 수
predict_interval_list = np.arange(0,24*7+1,6)
time_str = '2022/08/27/00UTC'   
shock = 1000
level = pres_list.index(str(500))   

surface_factors = []  # 예시: 지표면에서는 'MSLP'만 선택
upper_factors = ['q'] 


surface_factors.sort()
upper_factors.sort()
surface_str = "".join([f"_{factor}" for factor in surface_factors])  # 각 요소 앞에 _ 추가
upper_str = "".join([f"_{factor}" for factor in upper_factors])  # 각 요소 앞에 _ 추가


time_obj = datetime.strptime(time_str, "%Y/%m/%d/%HUTC")
final_time = time_obj + timedelta(hours=int(predict_interval_list[-1]))
final_str = final_time.strftime("%Y/%m/%d/%HUTC")
datetime_list = [time_obj + timedelta(hours=int(hours)) for hours in predict_interval_list]
# 
# storm_lon, storm_lat, storm_mslp, storm_time = storm_info(storm_name, storm_year, datetime_list = datetime_list)   #태풍 영문명, 년도 입력

min_position = []  # 태풍 중심 정보 dictionary



output_data_dir = rf'{pangu_dir}/output_data/{time_str}/{upper_str}SH_{shock}/{level}'
lon_space=20
lat_end
# :lon_end + 1], lat_indices[lat_start:lat_end + 1]
# int((lon_start+lon_end)/2)-lon_space:int((lon_start+lon_end)/2)+lon_space
#%%
for predict_interval in predict_interval_list:
    # print(predict_interval)
    predict_time = time_obj + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
    upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    mslp = met.met_data('MSLP')
    q = met.met_data('q', level = 500)
    print(q)
    wind_speed = met.wind_speed()
    z_diff = met.met_data('z', level = 200) - met.met_data('z', level = 500)
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{time_str} (+{predict_interval}h)\n{predict_str}', fontsize=20, loc = 'left')
    ax.set_title(f'SH{surface_str}{upper_str}_{shock}_{pres_list[level]}hPa Track\n{storm_name}', fontsize=20, loc = 'right')
    ax.set_extent(extent, crs=proj)
    setup_map(ax)

    # ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=3)
    # model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=3)
    # plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)

    pic = plt.contourf(lon_grid, lat_grid, q, levels = np.linspace(0,15,16), cmap='jet')
    plt.colorbar(pic, ax=ax, orientation='vertical', label='Specific Humidity(g/kg)', shrink=0.8)
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{time_str} (+{predict_interval}h)\n{predict_str}', fontsize=20, loc = 'left')
    ax.set_title(f'SH{surface_str}{upper_str}_{shock}_{pres_list[level]}hPa Track\n{storm_name}', fontsize=20, loc = 'right')
    ax.set_extent(extent, crs=proj)
    setup_map(ax)
    u = met.met_data('u', level = 500)
    plt.scatter(lon_grid, lat_grid,c=u, cmap = 'seismic', vmin = -100, vmax = 100)
    plt.show()
    
    lon_space = 20

    x = np.arange(int((lon_indices[lon_start]+lon_indices[lon_end])/2)-lon_space/4,int((lon_indices[lon_start]+lon_indices[lon_end])/2)+lon_space/4, 0.25)
    y = np.array(pres_list, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)
    q = met.met_data('q', level = 'all')
    q = q[:,int((lat_start+lat_end)/2),int((lon_start+lon_end)/2)-lon_space:int((lon_start+lon_end)/2)+lon_space]
    plt.contourf(xx, yy, q, cmap = 'jet', levels = np.linspace(0,20,41), extend = 'both')
    # plt.imshow(z[:,mp['idx'][0],mp['idx'][1]-lon_space:mp['idx'][1]+lon_space])
    plt.gca().invert_yaxis()

    cbar = plt.colorbar()
    cbar.set_label('Specific Humidity(g/kg)') 
    plt.ylabel('Pressure(hPa)')
    plt.xlabel('Lon(°)')
    plt.grid()
    plt.show()

    # min_position = plot_min_value(mslp, lat_indices, lon_indices, lat_start, lon_start, lat_grid, lon_grid, 
    #                                 wind_speed, predict_str, z_diff, storm_lon, storm_lat, storm_mslp, storm_time, 
    #                                 min_position, ax = None, mask_size = 15)

#%%
for predict_interval in predict_interval_list:
    # print(predict_interval)
    predict_time = time_obj + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
    upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
    met = Met(output_data_dir, predict_interval, surface_dict, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
    mslp = met.met_data('MSLP')
    q = met.met_data('q', level = 500)
    plt.imshow(q, vmin=0,vmax=10)
    plt.colorbar()
    plt.show()
    break
#%%
storm_lon = storm_lon[storm_time<=predict_time]
storm_lat = storm_lat[storm_time<=predict_time]
storm_mslp = storm_mslp[storm_time<=predict_time]
storm_time = storm_time[storm_time<=predict_time]

# for ens in range(ens_num):
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
ax.set_title(f'{time_str} (+{predict_interval_list[-1]}h)\n{predict_str}', fontsize=20, loc = 'left')
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

#%%
for ens in range(ens_num):
    lons = [pos['lon'] for pos in min_position[ens]]
    lats = [pos['lat'] for pos in min_position[ens]]
    min_values = [pos['mslp'] for pos in min_position[ens]]
    pred_times = [pos['time'] for pos in min_position[ens]]

    lc = colorline(ax, lons, lats, z=min_values, cmap=plt.get_cmap('jet_r'), norm=mcolors.Normalize(vmin=950, vmax=1020), linewidth=2, alpha=1.0)

    for i in range(len(pred_times)):
        if pred_times[i].endswith('00UTC'):
            ax.text(lons[i],lats[i], pred_times[i][8:-6]
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)

    if ens == 0:
        ax.text(lons[-1],lats[-1], '0 ENS'
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)


ax.legend(loc='upper right')
plt.show()



#%%
#그룹 찾기
lat_values = []

for ens in range(ens_num):
    for pos in min_position[ens]:
        if pos['time'] == '2022/08/30/00UTC':
            lat_values.append([ens, pos['lat']])  # (ens index, lat)

# lat_values
# 위도(lat) 값이 가장 높은 ens와 가장 낮은 ens를 찾음
# lat_values = np.array(lat_values)
lat_values.sort(key=lambda x: x[1])
group1 = [item[0] for item in lat_values[:10]]
group2 = [item[0] for item in lat_values[-10:]]



#%%
# group1 = []
# group2 = []
# for i in range(100):
#     for min_lon, min_lat, min_index, pred_str, min_value, error, _ in min_position[i]:
#         if (pred_str == '2022/09/02/00UTC') or (pred_str == '2022/09/01/00UTC'):
#             if (min_lon>=-50) & (min_lat>=30):
#                 group1.append(i)
                
#         if pred_str == '2022/09/03/00UTC':
#             if (min_lon<-50) & (min_lat<30):
#                 group2.append(i)
# # group3에 속하지 않은 숫자 지정
# all_indices = set(range(100))  # 전체 인덱스 집합
# group1_set = set(group1)  # group1 인덱스를 집합으로 변환
# group2_set = set(group2)  # group2 인덱스를 집합으로 변환
# # group1과 group2에 속하지 않은 인덱스를 찾아 group3으로 지정
# group3 = list(all_indices - group1_set - group2_set)
# group3_set = set(group3)
# print(group1_set & group3_set)
# print(len(group1), len(group2), len(group3))
#%%

from collections import defaultdict
group1_dict = {}
group2_dict = {}

group1_dict = defaultdict(lambda: {'lon': [], 'lat': [], 'value': []})
group2_dict = defaultdict(lambda: {'lon': [], 'lat': [], 'value': []})

for ens in group1:
    for min_lon, min_lat, min_index, pred_str, min_value, error, _ in min_position[ens]:
        group1_dict[pred_str]['lon'].append(min_lon)
        group1_dict[pred_str]['lat'].append(min_lat)
        group1_dict[pred_str]['value'].append(min_value)

for ens in group2:
    for min_lon, min_lat, min_index, pred_str, min_value, error, _ in min_position[ens]:
        group2_dict[pred_str]['lon'].append(min_lon)
        group2_dict[pred_str]['lat'].append(min_lat)
        group2_dict[pred_str]['value'].append(min_value)

# 각 pred_str에 대해 lon, lat, value의 평균 계산
for pred_str, values in group1_dict.items():
    lon_mean = np.mean(values['lon'])
    lat_mean = np.mean(values['lat'])
    value_mean = np.mean(values['value'])
    group1_dict[pred_str] = {'lon': lon_mean+180, 'lat': lat_mean, 'value': value_mean/100}

for pred_str, values in group2_dict.items():
    lon_mean = np.mean(values['lon'])
    lat_mean = np.mean(values['lat'])
    value_mean = np.mean(values['value'])
    group2_dict[pred_str] = {'lon': lon_mean+180, 'lat': lat_mean, 'value': value_mean/100}

# 결과 확인
for pred_str, values in group1_dict.items():
    print(f"{pred_str}: Lon: {values['lon']}, Lat: {values['lat']}, Value: {values['value']}")

for pred_str, values in group2_dict.items():
    print(f"{pred_str}: Lon: {values['lon']}, Lat: {values['lat']}, Value: {values['value']}")
    
fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
fig2, ax2 = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10))
ax.set_title(f'Group 1 & 2 Track', fontsize=30)
ax.set_extent(extent, crs=proj)
setup_map(ax)
ax2.set_title(f'Group 1 & 2 Intensity', fontsize=30)


for group_dict, marker, label in [(group1_dict, 'o', 'Group 1'), (group2_dict, '^', 'Group 2')]:
    sorted_keys = sorted(group_dict.keys(), key=lambda x: datetime.strptime(x, "%Y/%m/%d/%HUTC"))
    lons = [group_dict[key]['lon'] for key in sorted_keys]
    lats = [group_dict[key]['lat'] for key in sorted_keys]
    values = [group_dict[key]['value'] for key in sorted_keys]
    ax.plot(lons, lats, label=label, transform=ccrs.PlateCarree())
    print(sorted_keys)
    ax2.plot(sorted_keys,values, label=label)
    g12 = ax.scatter(lons, lats, c=values, marker=marker,cmap='jet_r',norm=norm_p, transform=ccrs.PlateCarree(), zorder=3, s=100)

plt.colorbar(g12, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=3)
model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=3)
# plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)
for i in range(len(storm_time)):
    new_time = storm_time[i].strftime("%Y/%m/%d/%HUTC")
    if new_time.endswith('00UTC'):
        dx, dy = 3, -3  # 시간 나타낼 위치 조정
        new_lon, new_lat = storm_lon[i] + dx, storm_lat[i] + dy
        
        # annotate를 사용하여 텍스트와 함께 선(화살표)을 그림
        ax.text(storm_lon[i], storm_lat[i], new_time[8:-6]
                , horizontalalignment='center', verticalalignment='bottom', fontsize=10)
ax.legend(loc='upper right', fontsize=15)
# ax2 설정
ax2.legend(loc='upper right', fontsize=15)
ax2.set_ylabel('Center Pressure(hPa)', fontsize=15)
ax2.tick_params(axis='both', which='major', labelsize=12)  # 틱 설정
ax2.grid(True)  # 그리드 추가

# 날짜 형식 설정
import matplotlib.dates as mdates
# ax2.xaxis.set_major_locator(mdates.DayLocator())

# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.show()
# ax.plot(storm_lon, storm_lat, color='black', linestyle='-', marker='', label = 'best track', transform=ax.projection, zorder=3)
# model_pred_sc = ax.scatter(storm_lon, storm_lat, c=storm_mslp, cmap='jet_r', marker='^',norm=norm_p, transform=ax.projection, zorder=3)
# plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='MSLP (hPa)', shrink=0.8)

#%%

levels = 5880
for predict_interval in predict_interval_list:
    
    group1_z_500_values = []
    group2_z_500_values = []
# for predict_interval in [120]:
    predict_time = time_obj + timedelta(hours=int(predict_interval))
    predict_str = predict_time.strftime("%Y/%m/%d/%HUTC")
    
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{predict_str}', fontsize=25)
    # Group 1
    for ens in group1:
        output_data_dir = rf'{pangu_dir}/output_data/{time_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
        # mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500 = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        # u_500 = upper[upper_dict['u'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
        # v_500 = upper[upper_dict['v'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
        z_500 = upper[upper_dict['z'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
        ax.contour(lon_grid, lat_grid, z_500, colors='red', levels = [levels], alpha=0.1)
        # z_850 = surface[surface_dict['T2M']][lat_start:lat_end + 1, lon_start:lon_end + 1]
        # v_850 = upper[upper_dict['v'], 2, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1]
        
        group1_z_500_values.append(z_500)
        # print(ens)


    # Group 2
    for ens in group2:
        output_data_dir = rf'{pangu_dir}/output_data/{time_str}/{perturation_scale}ENS{surface_str}{upper_str}/{ens}'
        surface = np.load(os.path.join(output_data_dir, rf'surface/{predict_interval}h.npy')).astype(np.float32)
        upper = np.load(os.path.join(output_data_dir, rf'upper/{predict_interval}h.npy')).astype(np.float32)
        # z_850 = surface[surface_dict['T2M']][lat_start:lat_end + 1, lon_start:lon_end + 1]
        z_500 = upper[upper_dict['z'], 5, :, :][lat_start:lat_end + 1, lon_start:lon_end + 1] / 9.80665
        ax.contour(lon_grid, lat_grid, z_500, colors='blue',  levels = [levels], alpha=0.1)
        # mslp, wind_speed, z_diff, z_500, vorticity, u_500, v_500 = process_meteorological_data(surface, surface_dict, upper, upper_dict, lat_start, lat_end, lon_start, lon_end, lat_grid, lon_grid)
        group2_z_500_values.append(z_500)
        # print(ens)
    
    # 각 그룹의 z_500 배열들을 하나의 큰 3D 배열로 결합
    group1_z_500_stack = np.stack(group1_z_500_values)
    group2_z_500_stack = np.stack(group2_z_500_values)

    print(np.shape(group1_z_500_stack))
    print(np.shape(group2_z_500_stack))

    # 각 지점에서의 z_500 값들의 평균 계산
    group1_z_500_mean = np.mean(group1_z_500_stack, axis=0)
    group2_z_500_mean = np.mean(group2_z_500_stack, axis=0)

    
    
    # ax.set_title(f'ENS{surface_str}{upper_str}{perturation_scale} Track\n{storm_name}', fontsize=20, loc = 'right')
    ax.set_extent(extent, crs=proj)
    setup_map(ax, back_color = 'n')
    g1 = ax.contour(lon_grid, lat_grid, group1_z_500_mean, colors='red',  levels = [levels])
    g2 = ax.contour(lon_grid, lat_grid, group2_z_500_mean, colors='blue',  levels = [levels])
    ax.clabel(g1, fmt='Group 1', inline=True, fontsize=20)
    ax.clabel(g2, fmt='Group 2', inline=True, fontsize=20)
    # model_pred_sc = ax.scatter(lon_grid, lat_grid, c=group1_z_500_mean-group2_z_500_mean, cmap='seismic',norm=mcolors.Normalize(vmin=-5, vmax=5), transform=ax.projection)
    # plt.colorbar(model_pred_sc, ax=ax, orientation='vertical', label='Temperature (K)', shrink=0.8)
    plt.show()