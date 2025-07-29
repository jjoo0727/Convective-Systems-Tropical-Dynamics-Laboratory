# Pangu Weather
폴더 구조는 다음과 같은 하위계층을 따른다.
```
├── root
│   ├── input_data
│   │   ├── YYYY.MM.DD HHUTC(multi_time의 경우 YYYY-MM-DD-HHUTC로 구성)
│   │   │   ├── surface.npy
│   │   │   ├── upper.npy
│   ├── output_data
│   │   ├── YYYY.MM.DD HHUTC(multi_time의 경우 YYYY-MM-DD-HHUTC로 구성)
│   │   │   ├── surface
│   │   │   │   ├── HHHh.npy
│   │   │   ├── upper
│   │   │   │   ├── HHHh.npy
│   ├── pangu_weather_1.onnx
│   ├── pangu_weather_3.onnx
│   ├── pangu_weather_6.onnx
│   ├── pangu_weather_24.onnx
│   ├── inference_cpu.py
│   ├── inference_gpu.py
│   ├── inference_iterative.py
```

cpu 이용시 cmd나 prompt에 다음을 입력(requirements_cpu.txt의 위치에서 실행해야 함)
```
pip install -r requirements_cpu.txt
```

GPU 이용시 CUDA 11.6 버전, Linux이면 8.2.4 버전, Window이면 cudnn 8.5.0.96 버전을 설치하고 다음을 실행
<br/>(CUDA가 굉장히 크므로 아직 다루지 않았음)
```
pip install -r requirements_gpu.txt
```

필요한 패키지 다운
```
pip install ai-models-panguweather
```

[Pangu Git](https://github.com/198808xc/Pangu-Weather?tab=readme-ov-file#downloading-trained-models)에 나온 onnx 파일들도 다운 받는다.

## ERA5_download.py
==원하는 ERA5 시간 데이터를 Pangu input 형식에 맞게 변형하는 코드==
<br/> ERA5 개인 url, key 필요
<br/> 각 변수들은 [Pangu Git](https://github.com/198808xc/Pangu-Weather?tab=readme-ov-file#downloading-trained-models)의 설명을 참조하라.
<br/> 시간대 조정은 다음과 같이 한다.
```python
time_str = 'YYYY.MM.DD HHUTC'
```

## inference_cpu.py, inference_iterative.py
time_str, predict_interval을 조정하여 초기장 시각과 예측 시간(1,3,6,24 중) 조정
<br/>inference_iterative.py에서는 예측 시간을 조합하여 더 먼 미래의 예측장 도출

## plotting.py
time_str, predict_interval_list을 조정하여 초기장 시각과 예측 시간 리스트 조정  
latlon_extent(0,360,-90,90)에서 살펴볼 위경도 범위 조정  
pres로 기압면 결정

이미지 예시
![image](https://github.com/jjoo0727/Convective-Systems-Tropical-Dynamics-Laboratory/assets/63052158/9dbac0d2-23b2-4d7e-9871-b21830442bb4)
![image](https://github.com/jjoo0727/Convective-Systems-Tropical-Dynamics-Laboratory/assets/63052158/f3971dc9-caa8-4c04-a992-e6dac09571b3)


## typhoon.py
==태풍 시작 위치와 시각을 입력하면 자동으로 추적하는 코드==
  
wind_mask는 지상풍 10m/s 이상인 지역만,  
expanded_wind_mask는 wind_mask 주변 2픽셀까지 포함
```python
wind_mask = wind_speed > 10
expanded_wind_mask = binary_dilation(wind_mask, structure=np.ones((5, 5)))
```


  
처음 찾을 때 init_pos(첫 위도, 첫 경도) 주변을 제외하 모두 마스킹
```python
if len(min_position) < 1:
  data_copy[(lat_grid > (init_pos[0]+init_size))|(lat_grid < (init_pos[0]-init_size))] = np.nan   
  data_copy[(lon_grid > (init_pos[1]+init_size-180))|(lon_grid < (init_pos[1]-init_size-180))] = np.nan 
```


vorticity 구할 때 lat가 내림차순으로 배치되어 있으므로  
du_dy를 거꾸로 연산해야 원하는 값이 도출됨됨  
```python
dv_dx = np.empty_like(v)
dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * delta_lon[:, 1:-1])
dv_dx[:, 0] = dv_dx[:, -1] = np.nan

du_dy = np.empty_like(u)
du_dy[1:-1, :] = (u[:-2, :] - u[2:, :]) / (2 * delta_lat[1:-1, :])
du_dy[0, :] = du_dy[-1, :] = np.nan
```
![image](https://github.com/jjoo0727/Convective-Systems-Tropical-Dynamics-Laboratory/assets/63052158/5a56867b-60a6-4ab7-9a98-0bfc666e5365)
![168h](https://github.com/jjoo0727/Convective-Systems-Tropical-Dynamics-Laboratory/assets/63052158/795f36a4-557e-46b2-a21b-df3117861a7f)

**현재 육상에 올라가면 정확한 위치 추적 힘듦**  

  

WMO 기준 35knot를 넘는 순간부터 추적
```python
def storm_info(storm_name, storm_year):
    ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)
    storm = ibtracs.get_storm((storm_name,storm_year))
    storm.to_dataframe()
    first_index = np.where((storm['wmo_vmax']>=35) == True)[0][0]
    storm_lon = storm['lon'][first_index:]-180
    storm_lat = storm['lat'][first_index:]
    storm_mslp = storm['mslp'][first_index:]
    storm_time = storm['time'][first_index:]
    return storm_lon, storm_lat, storm_mslp, storm_time
'''





      


