# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:40:06 2024

@author: jjoo0
"""
#%%
import cdsapi
import netCDF4 as nc
import numpy as np
import concurrent.futures
from datetime import datetime
import os
import itertools


def download_era5_surface(year, month, day, time, area, file_path):
    c = cdsapi.Client(
        # url="https://cds.climate.copernicus.eu/api/v2",
        url="https://cds.climate.copernicus.eu/api",
        # key="65047:f1c3f20d-b7ef-43c6-9a7c-540e8651e7fa")
        key="e8dc52f8-1ae5-4de1-a701-24218a59943a")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature'
            ],
            'year': year,
            'month': month,
            'day': day,
            'time': [i+':00' for i in time],
            'area': area,
        },
        file_path
    )

def download_era5_upper(year, month, day, time, area, file_path):
    c = cdsapi.Client(
        # url="https://cds.climate.copernicus.eu/api/v2",
        url="https://cds.climate.copernicus.eu/api",
        # key="65047:f1c3f20d-b7ef-43c6-9a7c-540e8651e7fa")
        key="e8dc52f8-1ae5-4de1-a701-24218a59943a")
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50'
            ],
            'year': year,
            'month': month,
            'day': day,
            'time': [i+':00' for i in time],
            'area': area,
        },
        file_path
    )



# 'YYYY.MM.DD HHUTC' 형식으로 적으면 됩니다!


# Parse the time string using datetime
# parsed_time = datetime.strptime(time_str, '%Y.%m.%d %HUTC')

# Extract year, month, day, and time
# year = str(parsed_time.year)
# month = '{:02d}'.format(parsed_time.month)
# day = '{:02d}'.format(parsed_time.day)
# time = '{:02d}:{:02d}'.format(parsed_time.hour, parsed_time.minute)
year = ['2012']
month = ['06']
# day = np.arange(1,32,1).astype(str)
day = ['22']
day = ['0' + i if len(i) < 2 else i for i in day]
times = ['00','12']
area = [90, 0, -90, 360]
time_len = len(year)*len(month)*len(day)*len(times)



download_time_str = f'{year[0]}.{month[0]}.{day[0]}_{times[0]}UTC_{year[-1]}.{month[-1]}.{day[-1]}_{times[-1]}UTC'
# Example file paths
upper_file_path = rf'/home1/jek/Pangu-Weather/input_data/download_upper_{download_time_str}.nc'
surface_file_path = rf'/home1/jek/Pangu-Weather/input_data/download_surface_{download_time_str}.nc'


# #donwload upper
download_era5_upper(year, month, day, times, area, upper_file_path)

dataset = nc.Dataset(upper_file_path)

time_step = 0
for y, m, d, tm in itertools.product(year, month, day, times):

    if d == '31' and m == '09':
        continue

    if not os.path.exists(rf'/home1/jek/Pangu-Weather/input_data/{y}/{m}/{d}/{tm}UTC'):
        os.makedirs(rf'/home1/jek/Pangu-Weather/input_data/{y}/{m}/{d}/{tm}UTC')
    z = np.array(dataset.variables['z'][:])[time_step]
    q = np.array(dataset.variables['q'][:])[time_step]
    t = np.array(dataset.variables['t'][:])[time_step]
    u = np.array(dataset.variables['u'][:])[time_step]
    v = np.array(dataset.variables['v'][:])[time_step]
    concatenated_data = np.stack([z, q, t, u, v], axis=0)
    time_step+=1
    np.save(rf'/home1/jek/Pangu-Weather/input_data/{y}/{m}/{d}/{tm}UTC/upper.npy', concatenated_data)


dataset.close()


download_era5_surface(year, month, day, times, area, surface_file_path)

dataset = nc.Dataset(surface_file_path)
time_step = 0
for y, m, d, tm in itertools.product(year, month, day, times):

    if d == '31' and m == '09':
        continue

    if not os.path.exists(rf'/home1/jek/Pangu-Weather/input_data/{y}/{m}/{d}/{tm}UTC'):
        os.makedirs(rf'/home1/jek/Pangu-Weather/input_data/{y}/{m}/{d}/{tm}UTC')
    msl = np.array(dataset.variables['msl'][:])[time_step]
    u10 = np.array(dataset.variables['u10'][:])[time_step]
    v10 = np.array(dataset.variables['v10'][:])[time_step]
    t2m = np.array(dataset.variables['t2m'][:])[time_step]
    concatenated_data = np.stack([msl, u10, v10, t2m], axis=0)
    time_step+=1
    np.save(rf'/home1/jek/Pangu-Weather/input_data/{y}/{m}/{d}/{tm}UTC/surface.npy', concatenated_data)

dataset.close()

#%%
import torch
print('CUDA version:', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())
print('CUDA available:', torch.cuda.is_available())