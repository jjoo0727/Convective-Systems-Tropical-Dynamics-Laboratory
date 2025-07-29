import os
import numpy as np
import onnx
import onnxruntime as ort
import time

start = time.time()

# The directory of your input and output data
predict_interval = 24                               #예측할 시간 지정(1,3,6,24) 가능
time_str = '2018.09.30 00UTC'                       #time_str을 조정해서 초기 시간 조정
pangu_dir = r'C:\Users\jjoo0\2023c\Pangu-Weather'   #각자 데스크톱에서 PANGU Weather 폴더 위치
input_data_dir = rf'{pangu_dir}\input_data'
output_data_dir = rf'{pangu_dir}\output_data'
model_dir = rf'{pangu_dir}\pangu_weather_{predict_interval}.onnx'
model = onnx.load(model_dir)

# 디렉토리 확인 및 생성
if not os.path.exists(rf'{output_data_dir}\upper'):
    os.makedirs(rf'{output_data_dir}\upper')
if not os.path.exists(rf'{output_data_dir}\surface'):
    os.makedirs(rf'{output_data_dir}\surface')

for predict_interval in [1,3,6,24]:                 #1,3,6,24시간 예측, 하나만 필요하면 for문 해제

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 10
    
    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}
    
    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session = ort.InferenceSession(model_dir, sess_options=options, providers=['CPUExecutionProvider'])
    
    # Load the upper-air numpy arrays
    input_upper = np.load(os.path.join(input_data_dir, rf'{time_str}\upper.npy')).astype(np.float32)
    # Load the surface numpy arrays
    input_surface = np.load(os.path.join(input_data_dir, rf'{time_str}\surface.npy')).astype(np.float32)
    
    # Run the inference session
    output, output_surface = ort_session.run(None, {'input':input_upper, 'input_surface':input_surface})
    
    # Save the results
    np.save(os.path.join(output_data_dir, rf'{time_str}\upper\{predict_interval}h'), output)
    np.save(os.path.join(output_data_dir, rf'{time_str}\surface\{predict_interval}h'), output_surface)
    end = time.time()
    print(f"걸린 시간: {end-start}s")