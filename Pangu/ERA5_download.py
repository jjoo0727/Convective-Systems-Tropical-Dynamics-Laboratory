# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:40:06 2024

@author: jjoo0
"""

import cdsapi
import netCDF4 as nc
import numpy as np
import concurrent.futures
from datetime import datetime
import os

def download_era5_surface(year, month, day, time, area, file_path):
    c = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api/v2",
        key="65047:f1c3f20d-b7ef-43c6-9a7c-540e8651e7fa")
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
            'day': [day],
            'time': [time],
            'area': area,
        },
        file_path
    )

def download_era5_upper(year, month, day, time, area, file_path):
    c = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api/v2",     
        key="65047:f1c3f20d-b7ef-43c6-9a7c-540e8651e7fa")   #개인 키이므로 
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
            'day': [day],
            'time': [time],
            'area': area,
        },
        file_path
    )



# 'YYYY.MM.DD HHUTC' 형식으로 적으면 됩니다!
time_str = '2018.09.30 00UTC'

# Parse the time string using datetime
parsed_time = datetime.strptime(time_str, '%Y.%m.%d %HUTC')

# Extract year, month, day, and time
year = str(parsed_time.year)
month = '{:02d}'.format(parsed_time.month)
day = '{:02d}'.format(parsed_time.day)
time = '{:02d}:{:02d}'.format(parsed_time.hour, parsed_time.minute)
area = [90, 0, -90, 360]

# Example file paths
surface_file_path = r'C:\Users\jjoo0\2023c\Pangu-Weather\input_data\download_surface'
upper_file_path = r'C:\Users\jjoo0\2023c\Pangu-Weather\input_data\download_upper'
surface_file_path = f'{surface_file_path}_{time_str}.nc'
upper_file_path = f'{upper_file_path}_{time_str}.nc'

if not os.path.exists(rf'C:\Users\jjoo0\2023c\Pangu-Weather\input_data\{time_str}'):
    os.makedirs(rf'C:\Users\jjoo0\2023c\Pangu-Weather\input_data\{time_str}')


download_era5_surface(year, month, day, time, area, surface_file_path)
download_era5_upper(year, month, day, time, area, upper_file_path)

dataset = nc.Dataset(upper_file_path)
z = np.array(dataset.variables['z'][:])
q = np.array(dataset.variables['q'][:])
t = np.array(dataset.variables['t'][:])
u = np.array(dataset.variables['u'][:])
v = np.array(dataset.variables['v'][:])
concatenated_data = np.concatenate([z, q, t, u, v], axis=0)

np.save(rf'C:\Users\jjoo0\2023c\Pangu-Weather\input_data\{time_str}\upper.npy', concatenated_data)
dataset.close()



dataset = nc.Dataset(surface_file_path)
msl = np.array(dataset.variables['msl'][:])
u10 = np.array(dataset.variables['u10'][:])
v10 = np.array(dataset.variables['v10'][:])
t2m = np.array(dataset.variables['t2m'][:])
concatenated_data = np.concatenate([msl, u10, v10, t2m], axis=0)

np.save(rf'C:\Users\jjoo0\2023c\Pangu-Weather\input_data\{time_str}\surface.npy', concatenated_data)
dataset.close()
#%%
# os.remove(upper_file_path)
# os.remove(surface_file_path)
