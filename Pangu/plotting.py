import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

lat_indices = np.linspace(90, -90, 721)
lon_indices = np.linspace(-180, 180, 1441)[:-1]

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



predict_interval_list = np.arange(0,24*7+1,24)[1:]  #볼 예측 시간 지정
time_str = '2018.09.30 00UTC'                       #time_str 지정
proj = ccrs.PlateCarree(central_longitude=180)
original_cmap = plt.get_cmap("BrBG")
truncated_BrBG = truncate_colormap(original_cmap, minval=0.35, maxval=1.0) #수증기 colormap 지정

proj = ccrs.PlateCarree(central_longitude=180)  # Set central_longitude to 180
#위경도 범위 지정
#동반구는 0~180, 서반구는 180~360
lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(0,360,-90,90)  

pres_list = ['1000','925','850','700','600','500','400','300','250','200','150','100','50']
pres=500                                            #살펴볼 기압면 결정
p=pres_list.index(str(pres))


for predict_interval in predict_interval_list:
    output_data_dir = rf'C:\Users\jjoo0\2023c\Pangu-Weather\output_data\{time_str}'
    
    # Load surface data
    surface = np.load(os.path.join(output_data_dir, rf'surface\{predict_interval}h.npy')).astype(np.float32) 
    surface_factor = ['MSLP', 'U10', 'V10', 'T2M']


    fig, axs = plt.subplots(2, 2, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    fig.suptitle(f'{time_str} Surface (+{predict_interval}h)', fontsize=36, weight='bold')
    
    # Plot the surface data on the map
    for i in range(4):
        ax = axs.flatten()[i]
        draw = surface[i][lat_start:lat_end + 1, lon_start:lon_end + 1]
    
        if surface_factor[i] == 'MSLP':
            im = ax.imshow(draw/100, cmap='seismic_r', extent = extent, transform=proj)
        elif surface_factor[i] == 'U10' or surface_factor[i] == 'V10':
            im = ax.imshow(draw, cmap='seismic', vmin = -50, vmax = 50, extent = extent,transform=proj)
        else:
            im = ax.imshow(draw, cmap='jet', extent = extent, transform=proj)
    
        # Add coastlines and gridlines
        ax.coastlines(zorder=1)
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 18}
        gl.ylabel_style = {'size': 18}
    
        ax.set_title(surface_factor[i], fontsize=36, weight='bold')
    
        # Add colorbar with a larger font size
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05,fraction=0.040)
        cbar.ax.tick_params(labelsize=18)
    
        # Set units for the colorbar
        if i == 0:
            cbar.set_label('Pressure (hPa)', fontsize=20)
        else:
            cbar.set_label('Temperature (K)' if i == 3 else 'Wind Speed (m/s)', fontsize=20)
    

    plt.tight_layout()
    plt.show()
    
    
    #wind vector map
    skip_step = 4   #skip_step이 클수록 표시되는 바람 벡터 감소
    u10 = surface[1][lat_start:lat_end + 1, lon_start:lon_end + 1][::skip_step, ::skip_step]
    v10 = surface[2][lat_start:lat_end + 1, lon_start:lon_end + 1][::skip_step, ::skip_step]
    fig, ax = plt.subplots(1, 1, figsize=(10*latlon_ratio, 10), subplot_kw={'projection': proj})
    ax.set_title(f'{time_str} Surface Wind Vector (+{predict_interval}h)', fontsize=30)
    
    quiver = ax.quiver(lon_indices[lon_start:lon_end + 1:skip_step], lat_indices[lat_start:lat_end + 1:skip_step], u10, v10, transform=proj
              , scale=500, color='black', headlength=4, headaxislength=2)   #scale이 클수록 작아짐, head를 통해 화살표 머리 크기 조정 가능
    
    land_color = (0.8, 0.9, 0.7) 
    ocean_color = (0.7, 0.9, 1.0)
    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color=land_color, edgecolor='black', zorder=0)
    ax.add_feature(cfeature.OCEAN, color=ocean_color, edgecolor='black', zorder=0)
    
    ax.quiverkey(quiver, X=0.95, Y=0.03, U=10, label='10 m/s', labelpos='E', color='r', labelcolor='r')
    ax.quiverkey(quiver, X=0.95, Y=0.06, U=1, label='1 m/s', labelpos='E', color='r', labelcolor='r')
    
    plt.show()
    
    
    # upper level 그리기
    upper = np.load(os.path.join(output_data_dir, rf'upper\{predict_interval}h.npy')).astype(np.float32) 
    upper_factor = ['Z', 'Q', 'T', 'U','V']
    upper_unit   = ['$m$', '$g/kg$', '$K$', '$m/s$', '$m/s$'] 
    

    # Create a figure and axis with a PlateCarree projection
    proj = ccrs.PlateCarree(central_longitude=180)
    
    lat_start, lat_end, lon_start, lon_end, extent, latlon_ratio = latlon_extent(110,160,5,45)

    fig, axs = plt.subplots(2, 3, figsize=(12*latlon_ratio*3/2, 12), subplot_kw={'projection': proj})
    fig.suptitle(f'{time_str} {pres}hPa (+{predict_interval}h)', fontsize=36, weight='bold')
    
    
    # Plot the surface data on the map
    for i in range(5):
        ax = axs.flatten()[i]
        draw = upper[i][p][lat_start:lat_end + 1, lon_start:lon_end + 1]
        
        if upper_factor[i] == 'Z':
            im = ax.imshow(draw/9.80665, cmap='seismic_r',extent = extent, transform=proj)
        elif upper_factor[i] == 'U' or upper_factor[i] == 'V':
            im = ax.imshow(draw, cmap='seismic', vmin = -50, vmax = 50, extent = extent,transform=proj)
        elif upper_factor[i] == 'Q':
            im = ax.imshow(draw*1000, cmap=truncated_BrBG, extent = extent,transform=proj)
        else:
            im = ax.imshow(draw, cmap='jet', extent = extent,transform=proj)

    
        # Add coastlines and gridlines
        ax.coastlines()
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 18}
        gl.ylabel_style = {'size': 18}
    
        ax.set_title(f'{upper_factor[i]}', fontsize=36, weight='bold')
    
        # Add colorbar with a larger font size
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, fraction=0.03)
        cbar.ax.tick_params(labelsize=18)
    
        # Set units for the colorbar
    
        cbar.set_label(f'{upper_unit[i]}', fontsize=20)
    
    fig.delaxes(axs[1][2])  #6번째 그림 없애기, 여기에 벡터 그림 추가하면 될 듯
    plt.tight_layout()
    plt.show()


