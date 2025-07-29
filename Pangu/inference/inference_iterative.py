#%%
# 여러 시점 동시에 돌리기
import os
import numpy as np
import onnx
import onnxruntime as ort
import time
import itertools

start = time.time()
year = ['2013']
month = ['11']
day = ['01','02','03']
times = ['00','06','12','18']
pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'

model_24 = onnx.load(rf'{pangu_dir}/pangu_weather_24.onnx')
model_6 = onnx.load(rf'{pangu_dir}/pangu_weather_6.onnx')


# Set the behavior of onnxruntime
options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False

# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 50

# Set the behavior of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
# ort_session_24 = ort.InferenceSession(rf'{pangu_dir}/pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
# ort_session_6 = ort.InferenceSession(rf'{pangu_dir}/pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

ort_session_24 = ort.InferenceSession(rf'{pangu_dir}/pangu_weather_24.onnx', sess_options=options, providers=['CPUExecutionProvider'])
ort_session_6 = ort.InferenceSession(rf'{pangu_dir}/pangu_weather_6.onnx', sess_options=options, providers=['CPUExecutionProvider'])



time_step=0
time_len = len(year)*len(month)*len(day)*len(times)
# Run the inference session
for y, m, d, tm in itertools.product(year, month, day, times):
    time_str = f'{y}/{m}/{d}/{tm}UTC'

    input_data_dir = rf'{pangu_dir}/input_data/{time_str}'
    output_data_dir = rf'{pangu_dir}/output_data/{time_str}'

    input_upper = np.load(os.path.join(input_data_dir, 'upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'surface.npy')).astype(np.float32)
    


    if not os.path.exists(os.path.join(output_data_dir, f'upper')):
        os.makedirs(os.path.join(output_data_dir, f'upper'))
    if not os.path.exists(os.path.join(output_data_dir, f'surface')):
        os.makedirs(os.path.join(output_data_dir, f'surface'))

    input_24, input_surface_24 = input_upper, input_surface

    for i in range(28):
        start_i = time.time()
        predict_interval = 6*(i+1)
        if (i+1) % 4 == 0:
            output, output_surface = ort_session_24.run(None, {'input':input_24, 'input_surface':input_surface_24})
            input_24, input_surface_24 = output, output_surface
            np.save(os.path.join(output_data_dir, f'upper/{predict_interval}h'), output)
            np.save(os.path.join(output_data_dir, f'surface/{predict_interval}h'), output_surface)


        # 6시간 간격도 저장하고 싶으면 주석 해제
        else:
            output, output_surface = ort_session_6.run(None, {'input':input_upper, 'input_surface':input_surface})
            np.save(os.path.join(output_data_dir, f'upper/{predict_interval}h'), output)
            np.save(os.path.join(output_data_dir, f'surface/{predict_interval}h'), output_surface)

        input_upper, input_surface = output, output_surface
        end_i = time.time()
        print(f'{i+1}번째 반복 +{predict_interval}h {end_i-start_i}s')

    end = time.time()
    print(f"총 걸린 시간: {end-start}s")
    time_step+=1
                            


