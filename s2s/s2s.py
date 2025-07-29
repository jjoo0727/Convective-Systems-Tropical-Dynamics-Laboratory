
#%%
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ty_pkg import truncate_colormap, colorline, setup_map, weather_map_contour, contourf_and_save, ep_t, concentric_circles, interpolate_data
from ty_pkg import latlon_extent, storm_info, haversine_distance, Met, calculate_bearing_position, tc_finder

import tarfile
import os
import shutil

from datetime import datetime
from datetime import timedelta

import pickle       

with open(r'/home1/jek/Pangu-Weather/code/s2s/data/wmo_data.pkl', 'rb') as file:
    wmo_data = pickle.load(file)
    
with open(r'/home1/jek/Pangu-Weather/code/s2s/data/jtwc_data.pkl', 'rb') as file:
    jtwc_data = pickle.load(file)


basin_list = ['atl', 'enp' , 'cnp' , 'wnp' , 'nin' , 'sin', 'aus', 'spc' ]

best_dict = {}

def print_char_codes(line):
    # 각 문자와 해당 ASCII/Unicode 값을 출력
    for char in line:
        print(f"'{char}' : {ord(char)}")

def reduce_spaces(text):
    # 여러 개의 공백을 한 개의 공백으로 치환
    return re.sub(r'\s+', ' ', text)


def clean_text(text):
    # NULL 문자 제거
    cleaned_text = text.replace('\x00', '')
    # 연속된 공백을 하나의 공백으로 치환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text



# Reading the file and parsing the data
def read_tc_file(file_path, best_dict):
    with open(file_path, 'r') as file:
        dict_key = ''
        valid_ty = False

        best_dict = {
                    'time': [],
                    'lat': [],
                    'lon': [],
                    'wind': [],
                    'mslp': []
                }
        
        for line in file:
            line = line.strip()
            line = clean_text(line)

            head_pattern = r"(\d{5}) (\d{2}/\d{2}/\d{4}) M=\s*(\d+) (\d+) SNBR=\s*(\d+)"
            pattern = r"(\d{5}) (\d{4}/\d{2}/\d{2}/\d{2})\*([\s\d]{3})([\s\d]{4}) +(\d{2}) +(\d{3,4})\*"
            head_match = re.search(head_pattern, line)
            match = re.search(pattern, line)

            
            
            
            
            if head_match:
                valid_ty = False  # Reset the flag for each new header
                
            if match:
                # 내가 지정한 날짜와 위치가 아닌 것들은 아예 best dict에서 제거
                if not valid_ty:
                    match_date = match.group(2)
                    if match_date == next_date:
                        if ((int(match.group(3)) / 10.0) >= target_lat - 3) & ((int(match.group(3)) / 10.0) <= target_lat + 3):
                            if ((int(match.group(4)) / 10.0) >= target_lon- 3) & ((int(match.group(4)) / 10.0) <= target_lon + 3):
                                valid_ty = True

                                # print(match_date, next_date)

                    else:
                        continue  # Skip processing this line
                
                # valid_ty에 120시간 예측까지만 추가                
                if valid_ty:
                    if datetime.strptime(match.group(2),"%Y/%m/%d/%H") <= target_datetime+timedelta(days=5):
                        best_dict['time'].append(match.group(2))
                        best_dict['lat'].append(int(match.group(3)) / 10.0)
                        best_dict['lon'].append(int(match.group(4)) / 10.0)
                        best_dict['wind'].append(int(match.group(5)))
                        best_dict['mslp'].append(int(match.group(6)))
    
    for key in best_dict:
        best_dict[key] = np.array(best_dict[key])         
    return best_dict            


# Plotting function
def plot_cyclone_track(data):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(0))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    for point in data:
        ax.plot(point['longitude'], point['latitude'], 'ro', transform=ccrs.Geodetic())
        ax.text(point['longitude'] + 0.5, point['latitude'], f"{point['datetime']}", transform=ccrs.Geodetic())

    plt.show()
#%%


target_date = '20220825'
target_datetime = datetime.strptime(target_date,"%Y%m%d")
next_datetime = datetime.strptime(target_date,"%Y%m%d") + timedelta(days = 1)
next_date = next_datetime.strftime("%Y/%m/%d/%H")
lonlat = {'lon':[], 'lat':[]}

for id, wmo in wmo_data.items():
    if id.startswith('WP') and wmo['name'] != 'UNNAMED':
        if next_datetime in wmo['time']:
            time_idx = wmo['time'].index(next_datetime)
            target_lon = wmo['lon'][time_idx]
            target_lat = wmo['lat'][time_idx]
        
        for i in range(1,6):
            timetime = datetime.strptime(target_date,"%Y%m%d") + timedelta(days = i)
            if timetime in wmo['time']:
                time_idx = wmo['time'].index(timetime)
                lonlat['lon'].append(wmo['lon'][time_idx])
                lonlat['lat'].append(wmo['lat'][time_idx])
                
lonlat['lon'], lonlat['lat'] = np.array(lonlat['lon']), np.array(lonlat['lat'])


# Step 1: Extract the initial tar file
initial_tar_file = f'/home1/jek/Pangu-Weather/code/s2s/data/TC.{target_date}'
with tarfile.open(initial_tar_file, 'r') as tar:
    tar.extractall()

# Create a directory to store wnp files
wnp_directory = f'/home1/jek/Pangu-Weather/code/s2s/data/{target_date}'
os.makedirs(wnp_directory, exist_ok=True)


# Step 2: Loop through each extracted file and extract it again
for file in os.listdir('.'):
    if file.startswith('ecmf.'):
        with tarfile.open(file, 'r') as tar:
            tar.extractall()
            # Find the wnp file
            for member in tar.getmembers():
                if 'wnp' in member.name:
                    # Step 3: Move the wnp file to the wnp_files directory
                    extracted_wnp_path = member.name
                    destination_path = os.path.join(wnp_directory, f'{file.split(".")[-1]}')
                    shutil.move(extracted_wnp_path, destination_path)
        # Optionally, remove the original file to clean up
        os.remove(file)
        

# Create a figure with PlateCarree projection
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree(0)})
setup_map(ax, back_color='n')
dis_error = []
for i in range(0,51,1):
    # Example usage
    file_path = rf"/home1/jek/Pangu-Weather/code/s2s/data/{target_date}/{i}"
    best_dict = read_tc_file(file_path, best_dict)

    # ax.set_extent([-180, 180, -90, 90])
    ax.set_extent([110, 160, 5, 45])

    # 부족한 길이만큼 NaN 값을 추가합니다.
    if len(best_dict['lon']) < len(lonlat['lon']):
        # 부족한 요소 수를 계산합니다.
        n_missing = len(lonlat['lon']) - len(best_dict['lon'])
        
        # 부족한 요소를 nan으로 채운 배열을 생성합니다.
        nan_fill = np.full(n_missing, np.nan)
        
        # 원래 배열과 새로 생성한 nan 배열을 연결합니다.
        best_dict['lon'] = np.concatenate((best_dict['lon'], nan_fill))
        best_dict['lat'] = np.concatenate((best_dict['lat'], nan_fill))
        
    print(lonlat['lon'], best_dict['lon'], lonlat['lat'], best_dict['lat'])
    
    lc = colorline(ax, best_dict['lon'], best_dict['lat'], z = best_dict['mslp'], norm = mcolors.Normalize(vmin=920, vmax=1020), linewidth=0.5)
    dis_error.append(haversine_distance(lonlat['lat'], lonlat['lon'], best_dict['lat'], best_dict['lon']))
        # ax.plot(best_dict['lon'], best_dict['lat'], linewidth=1, colorline = best_dict['mslp'])  # Simplified plot call
        # scatter = ax.scatter(best_dict['lon'], best_dict['lat'], s=2, c=best_dict['mslp'], cmap='gist_rainbow', norm=mcolors.Normalize(vmin=920, vmax=1020))  # Corrected color map and normalization
dis_error = np.array(dis_error)
dis_mean_error = np.nanmean(np.array(dis_error), axis = 0)    
# Add a color bar
cbar = fig.colorbar(lc, ax=ax, orientation='vertical', shrink = 0.8)
cbar.set_label('Mean Sea-Level Pressure (hPa)')

plt.show()

for i in range(5):
    nan_error = np.count_nonzero(~np.isnan(dis_error[:, i]))
    print(f"Non-NaN count in column {i}: {nan_error}")
dis_mean_error
