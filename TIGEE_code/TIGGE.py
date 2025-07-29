#%%
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from datetime import timedelta
import pickle   
import numpy as np
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns

import netCDF4 as nc

import time

with open(r'/home1/jek/Pangu-Weather/code/s2s/data/wmo_data.pkl', 'rb') as file:
    wmo_data = pickle.load(file)
    
with open(r'/home1/jek/Pangu-Weather/code/s2s/data/jtwc_data.pkl', 'rb') as file:
    jtwc_data = pickle.load(file)
    
    
@jit(nopython=True)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def ensemble_data_maker(filename):
    # XML 파일 파싱
    tree = ET.parse(f'{filename}.xml')
    root = tree.getroot()

    # 모든 'data' 요소에 대하여 처리
    ensemble_data = {}
    for data in root.findall('.//data'):
        if data.get('member') == None:
            continue
        member = int(data.get('member'))
        # print(member)
        perturb = data.get('perturb')

        # 멤버 별로 데이터 구성
        if member not in ensemble_data:
            ensemble_data[member] = {}

        # 각 'disturbance' 요소 처리
        for disturbance in data.findall('.//disturbance'):
            disturbance_id = disturbance.get('ID')
            basin = disturbance.find('basin').text
            
            if disturbance_id.endswith('00E'):
                continue
            
            # Basin이 Northwest Pacific인 경우만 데이터 저장
            if basin == "Northwest Pacific":
                
                # disturbance ID 별로 데이터 구성
                if disturbance_id not in ensemble_data[member]:
                    ensemble_data[member][disturbance_id] = {}

                # 각 'fix' 요소에서 위치와 기압 데이터 추출
                for fix in disturbance.findall('fix'):
                    valid_time = fix.find('validTime').text
                    valid_time = datetime.strptime(valid_time,"%Y-%m-%dT%H:%M:%SZ")
                    latitude = float(fix.find('latitude').text)
                    longitude = float(fix.find('longitude').text)
                    pressure = float(fix.find('.//pressure').text)
                    
                    ensemble_data[member][disturbance_id][valid_time] = {
                        'lon': longitude, 'lat': latitude, 'pres': pressure
                    }
    return ensemble_data

def nc_dict(filename):
    with nc.Dataset(f'{filename}.nc', 'r') as dataset:
        ensemble_data = {}
        for member_name in dataset.groups:
            member = int(member_name.split('_')[1])  # Extract member number
            ensemble_data[member] = {}
            member_group = dataset.groups[member_name]
            
            for disturbance_id in member_group.groups:
                disturbance_group = member_group.groups[disturbance_id]
                ensemble_data[member][disturbance_id] = {}
                
                times_var = disturbance_group.variables['time']
                if hasattr(times_var, 'units') and hasattr(times_var, 'calendar'):
                    # Converting time data
                    times = nc.num2date(times_var[:], units=times_var.units, calendar=times_var.calendar)
                    # Ensure all times are standard datetime objects
                    times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in times]
                else:
                    raise ValueError("Time variable is missing 'units' or 'calendar' attributes")

                latitudes = disturbance_group.variables['lat'][:]
                longitudes = disturbance_group.variables['lon'][:]
                pressures = disturbance_group.variables['pres'][:]
                
                for i, t in enumerate(times):
                    ensemble_data[member][disturbance_id][t] = {
                        'lat': latitudes[i], 'lon': longitudes[i], 'pres': pressures[i]
                    }
    return ensemble_data


# @jit(nopython=True)
# def convert_times(times, base_time):
#     # This function assumes times are in hours since 1970-01-01 00:00:00
#     converted_times = np.empty(len(times), dtype=np.object_)
#     for i in range(len(times)):
#         t = base_time + timedelta(hours=times[i])
#         converted_times[i] = datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
#     return converted_times

# def nc_dict(filename):
#     with nc.Dataset(f'{filename}.nc', 'r') as dataset:
#         ensemble_data = {}
#         for member_name in dataset.groups:
#             member = int(member_name.split('_')[1])  # Extract member number
#             ensemble_data[member] = {}
#             member_group = dataset.groups[member_name]
            
#             for disturbance_id in member_group.groups:
#                 disturbance_group = member_group.groups[disturbance_id]
#                 ensemble_data[member][disturbance_id] = {}
                
#                 times_var = disturbance_group.variables['time']
#                 if hasattr(times_var, 'units') and hasattr(times_var, 'calendar'):
#                     # Converting time data using Numba accelerated function
#                     base_time = datetime(1970, 1, 1)
#                     times = convert_times(times_var[:], base_time)
#                 else:
#                     raise ValueError("Time variable is missing 'units' or 'calendar' attributes")

#                 latitudes = disturbance_group.variables['lat'][:]
#                 longitudes = disturbance_group.variables['lon'][:]
#                 pressures = disturbance_group.variables['pres'][:]
                
#                 for i, t in enumerate(times):
#                     ensemble_data[member][disturbance_id][t] = {
#                         'lat': latitudes[i], 'lon': longitudes[i], 'pres': pressures[i]
#                     }
#     return ensemble_data
#%%
type(wmo_data['WP182023']['year'])
wmo_data['WP122022']['time']

#%%
# wmo_data_2007 = {i: v for i, v in wmo_data.items() if 'west_pacific' in v['wmo_basin']} #2007년 이후 데이터만 사용
wmo_data_2007 = {i: v for i, v in wmo_data.items() if ('west_pacific' in v['wmo_basin']) and (v['year'] >= 2009) and (v['year'] <= 2023)} #2007년 이후 데이터만 사용

time_all  = 0
time_file = 0
time_cal  = 0
overall_dis = {}

time_all_start = time.time()
for tc in wmo_data_2007:
    # print(tc)
    t_list = wmo_data_2007[tc]['time']
    lon_list = wmo_data_2007[tc]['lon']
    lat_list = wmo_data_2007[tc]['lat']
    
    # filtering datatimes end with 00 or 12UTC
    filtered_indices = [i for i, t in enumerate(t_list) if (t.hour == 12) or (t.hour == 0)]
    t_list   = [t_list[i] for i in filtered_indices]
    lon_list = [lon_list[i] for i in filtered_indices]
    lat_list = [lat_list[i] for i in filtered_indices]
    
    #gathering lon, lat in the t_list
    for i, t1 in enumerate(t_list):
        lon1 = lon_list[i]
        lat1 = lat_list[i]
        
        target_date = t1.strftime("%Y%m%d%H")
        filename = f"/home1/jek/Pangu-Weather/input_data/TIGGE/ecmf/{target_date[:4]}/{target_date[:8]}/z_tigge_c_ecmf_{target_date}0000_ifs_glob_test_all_glo"
        filename_prod = f"/home1/jek/Pangu-Weather/input_data/TIGGE/ecmf/{target_date[:4]}/{target_date[:8]}/z_tigge_c_ecmf_{target_date}0000_ifs_glob_prod_all_glo"
        # /home1/jek/Pangu-Weather/input_data/TIGGE/ecmf/2022/20220901/z_tigge_c_ecmf_20220901000000_ifs_glob_prod_all_glo.xml
        #bring TIGGE data if it exits
        time_file_start = time.time()
        try:
            ensemble_data = nc_dict(filename)
            
        except:
            try:
                ensemble_data = nc_dict(filename_prod)
            except:
                print(f'no file {filename}')
                continue
        
        
        time_file_end = time.time()
        time_file += (time_file_end - time_file_start)
        
        time_cal_start = time.time()
        #Make predict time list and get distance error
        dis = {}
        for t_delta in range(0,121,12):
            dis[t_delta] = []
            t2 = t1 + timedelta(hours = t_delta)
            
            # lon, lat of forecasted location
            if t2 in t_list:
                # print(1)
                lon2 = lon_list[t_list.index(t2)]
                lat2 = lat_list[t_list.index(t2)]
                
                mean_ens_error = []
                for member in ensemble_data:
                    for id in ensemble_data[member]:
                        if t1 in ensemble_data[member][id] and t2 in ensemble_data[member][id]:
                            tig_lon1, tig_lat1 = ensemble_data[member][id][t1]['lon'], ensemble_data[member][id][t1]['lat']
                            if (np.abs(tig_lon1-lon1)<2) & (np.abs(tig_lat1-lat1)<2):   
                                # print(1)
                                tig_lon2, tig_lat2 = ensemble_data[member][id][t2]['lon'], ensemble_data[member][id][t2]['lat']
                                distance_error = haversine_distance(tig_lat2, tig_lon2, lat2, lon2)
                                mean_ens_error.append(distance_error)
                                
                               
                                if t_delta == 12:
                                    if distance_error > 1000:
                                        print(tc, t1, dis[t_delta])
                                        print((lon1, lat1), (tig_lon1, tig_lat1))
                                        print((lon2, lat2), (tig_lon2, tig_lat2))
                                        
                dis[t_delta].append(mean_ens_error)
                
        for t_delta in dis:
            # if len(dis[t_delta]) >= 30:
            overall_dis.setdefault(t_delta, []).extend([dis[t_delta]])
        
        time_cal_end = time.time()
        time_cal += (time_cal_end - time_cal_start)
                    
time_all_end = time.time()              
time_all = time_all_end - time_all_start   
print(f"Total time: {time_all}s, Opeing time: {time_file}s, Calculate time: {time_cal}s")   

#%%
modified_dis = {}
for key in overall_dis:
    # print(overall_dis[key].keys())
    modified_dis[key] = []
    for mem in overall_dis[key]:
        if mem and mem != [[]]:
            modified_dis[key].append(np.array(mem[0]))

    # modified_dis[key] = np.array(modified_dis[key])
    
mean_dis = {}
for key in modified_dis:
    mean_dis[key] = []
    for mem in modified_dis[key]:
        mean_dis[key].append(np.mean(mem))
        
fig, ax = plt.subplots()
ax.boxplot(mean_dis.values(), labels=mean_dis.keys())

ax.scatter(np.array(list(dis.keys()))/12+1, dis.values(), color = 'red')
# ax.set_title('Box plot of Mean Values')
ax.set_xlabel('Prediction Time(h)')
ax.set_ylabel('Distance(km)')

plt.show()




#%%

datatime = datetime(2022,8,28,12,0)
filename_prod = f"/home1/jek/Pangu-Weather/input_data/TIGGE/ecmf/2022/20220828/z_tigge_c_ecmf_{datatime.strftime('%Y%m%d%H')}0000_ifs_glob_prod_all_glo"  
ensemble_data = nc_dict(filename_prod)


t_list = wmo_data['WP122022']['time']
lon_list = wmo_data['WP122022']['lon']
lat_list = wmo_data['WP122022']['lat']

# filtering datatimes end with 00 or 12UTC
filtered_indices = [i for i, t in enumerate(t_list) if (t.hour == 12) or (t.hour == 0)]
t_list   = [t_list[i] for i in filtered_indices]
lon_list = [lon_list[i] for i in filtered_indices]
lat_list = [lat_list[i] for i in filtered_indices]


dis = {}

for t_delta in range(0,121,12):
    t2 = datatime + timedelta(hours = t_delta)
    print(t2)
    # lon, lat of forecasted location
    if t2 in t_list:
        print(t2)
        print(t_delta)
        lon2 = lon_list[t_list.index(t2)]
        lat2 = lat_list[t_list.index(t2)]
        
        
        mean_ens_error = []
        for member in ensemble_data:
            # print(member)
            for id in ensemble_data[member]:
                if t2 in ensemble_data[member][id]:
                    # tig_lon1, tig_lat1 = ensemble_data[member][id][t2]['lon'], ensemble_data[member][id][t2]['lat']
                    # if (np.abs(tig_lon1-lon2)<2) & (np.abs(tig_lat1-lat2)<2):   
                    # print(member)
                    tig_lon2, tig_lat2 = ensemble_data[member][id][t2]['lon'], ensemble_data[member][id][t2]['lat']
                    distance_error = haversine_distance(tig_lat2, tig_lon2, lat2, lon2)
                    print(tig_lon2, tig_lat2, lon2,lat2,distance_error)
                    # print(distance_error)
                    mean_ens_error.append(distance_error)

        dis[t_delta] = np.mean(mean_ens_error)

dis
#%%
k = 0
for i in range(50):
    if len(ensemble_data[i]['2022082800_249N_1503E'])/4 < 4:
        k+=1
k
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# modified_dis 딕셔너리를 pandas DataFrame으로 변환
data = []
for t_delta, distances in modified_dis.items():
    for distance in distances:
        data.append({'Time Delta': t_delta, 'Distance': distance})
df = pd.DataFrame(data)

# Time Delta를 문자열로 변환하여 순서를 보장
df['Time Delta'] = df['Time Delta'].astype(str)

# 박스플롯 생성
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Time Delta', y='Distance', data=df)

# dis 딕셔너리의 각 값을 해당 Time Delta에 'x' 마크로 표시
# x축의 위치를 문자열로 변환된 Time Delta 값으로 찾음
time_delta_order = df['Time Delta'].unique()  # 박스플롯의 x축 순서
for t_delta, distance in dis.items():
    # 문자열 변환된 Time Delta 위치 찾기
    pos = np.where(time_delta_order == str(t_delta))[0]
    if pos.size > 0:
        plt.scatter(pos, distance, color='red', marker='x', s=100, zorder = 10)

plt.title('Distance Error Boxplot by Time Delta with Points')
# plt.xticks(rotation=45)  # x축 레이블 회전
plt.show()

#%%
import cartopy.crs as ccrs
from ty_pkg import colorline
filename_prod = f"/home1/jek/Pangu-Weather/input_data/TIGGE/ecmf/2022/20220828/z_tigge_c_ecmf_20220828120000_ifs_glob_prod_all_glo"  
ensemble_data = nc_dict(filename_prod)

# storm_key = '2022082800_249N_1503E'
storm_key = '2022082812_269N_1485E'
# for t, i in ensemble_data[0]['2022082800_249N_1503E'].items():

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

for ens in range(50):
    lat_list = [v['lat'] for i,v in ensemble_data[ens][storm_key].items()]
    lon_list = [v['lon'] for i,v in ensemble_data[ens][storm_key].items()]
    pres_list = [v['pres'] for i,v in ensemble_data[ens][storm_key].items()]
    time_list = [i for i,v in ensemble_data[ens][storm_key].items()]
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.005, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    # ax.plot(lon_list, lat_list, zorder = 1)
    ax.scatter(lon_list, lat_list, zorder = 2, c = pres_list, cmap = 'jet_r', s = 10)
    # colorline(ax, lon_list, lat_list, z = pres_list)
#%%
ensemble_data[0]['2022082800_249N_1503E']
#%%
#Histogram
# Set the style of seaborn for better visibility
sns.set(style="whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(nrows=len(overall_dis), ncols=1, figsize=(10, 5 * len(overall_dis)))

# Check if there's only one subplot (which does not return an array)
if len(overall_dis) == 1:
    axes = [axes]

# Loop through each time delta and plot a histogram
for ax, (t_delta, distances) in zip(axes, overall_dis.items()):
    sns.histplot(distances, binwidth = 100, kde=True, ax=ax)  # KDE=True adds a Kernel Density Estimate line
    ax.set_title(f"Time Delta {t_delta} hours")
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,3000)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#%%
#Box plot
# Set the style
sns.set(style="whitegrid")

# Create a single figure to host all boxplots
plt.figure(figsize=(12, 8))

# Data needs to be restructured for seaborn boxplot, using a DataFrame
import pandas as pd
data = []
for t_delta, distances in overall_dis.items():
    for distance in distances:
        data.append({'Time Delta': t_delta, 'Distance': distance})

df = pd.DataFrame(data)

# Create a boxplot
sns.boxplot(x='Time Delta', y='Distance', data=df)
plt.title('Distance Distribution by Time Delta')
plt.xlabel('Time Delta (hours)')
plt.ylabel('Distance (km)')

# Show the plot
plt.show()

#%%

import numpy as np
from scipy.stats import ttest_ind

# 두 집단의 데이터 샘플
hin_120 = [903.6937688871632, 1295.7888626211331, 866.9227684810998, 1028.6253258621653, 779.5268534958357, 606.9534364984268, 1221.831370267449, 1673.630251271355, 844.578792973374, 668.2400316211844, 859.5063923061485, 1028.6253258621653, 1123.8527025863132, 1789.6866566611577, 609.4052021329263, 1350.2409154156207, 1367.8908299938842, 1417.405992568515, 1075.0809860443633, 536.019787861839, 556.5445783127875, 2024.146771379165, 481.9403014740785, 881.4396603394148, 668.2400316211844, 1263.652091984331, 886.6360907620258, 615.4199395899061, 1038.2891554881426, 1719.7603657449563, 1529.150736281875, 1028.6253258621653, 1482.0870964770797, 1182.4533270646955, 652.5703687493958, 1263.520532371264, 1554.5474028963094, 829.7261668395267, 850.2062493375782, 1206.0163964459277, 964.584141125434, 674.5586812247313, 779.5268534958357, 1347.4859663813572, 1028.6253258621653, 1417.405992568515, 964.584141125434, 1227.0991201705901, 1204.6959672080645, 481.9403014740785, 802.8169926873635, 569.6993114321384, 812.5988131654441, 943.9036902255824, 964.584141125434, 949.0739055543619, 754.3647529507592, 774.4332731155683, 758.7943914801352, 1321.417525195511, 807.6970329438293, 471.32998292316057, 812.5988131654441, 822.5654053697389, 609.4052021329263, 1322.382370172114, 369.88551373899173, 1729.8213098561118, 457.7420025858192, 1081.5232181728989, 1197.0096810146676, 1175.312155596378, 733.8839344838482, 1322.382370172114, 646.1614420464479, 1123.8527025863132, 1160.4475062780261, 652.5703687493958, 550.5390499217972, 991.8717016047909, 1745.5765622983288, 668.2400316211844, 851.4691197015118, 830.1373144024121, 569.6993114321384, 1038.2891554881426, 822.5654053697389, 995.4836894598617, 736.2226160618162, 718.5055954031604, 503.86070048005803]
data_group1 = overall_dis[120]  # 평균 50, 표준편차 10, 데이터 개수 100
data_group2 = hin_120  # 평균 55, 다른 집단

# 독립표본 T 검정 수행
t_stat, p_value = ttest_ind(data_group1, data_group2, equal_var=False)

print("T-Statistic:", t_stat)
print("P-value:", p_value)


#%%
overall_dis
with open('/home1/jek/Pangu-Weather/input_data/TIGGE/dis_error.pkl', 'wb') as tf:
	pickle.dump(overall_dis, tf)
# overall_dis[120]


#%%
with open('/home1/jek/Pangu-Weather/input_data/TIGGE/dis_error.pkl', 'rb') as tf:
	overall_dis = pickle.load(tf)
 
overall_dis[120]