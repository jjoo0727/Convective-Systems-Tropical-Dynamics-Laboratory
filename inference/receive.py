
#%%
import subprocess
import os

year = '2022'
month = '08'
day = '27'
times = '00'
perturbation_scale = 0.05
factor_str = 'z'
pangu_dir = r'/Data/home/jjoo0727/Pangu-Weather'
time_str = f'{year}/{month}/{day}/{times}UTC'
rclone_path = "/home1/jek/rclone-v1.66.0-linux-amd64/rclone-v1.66.0-linux-amd64/rclone"  # rclone의 전체 경로

# 원격 서버 정보 및 로컬 경로 설정
remote_server = "ai_server"
local_server = "tcml"
remote_path = f"{pangu_dir}/output_data/{time_str}/{perturbation_scale}ENS_{factor_str}"
local_path = f"/home1/jek/Pangu-Weather/output_data/{time_str}/{perturbation_scale}ENS_{factor_str}"

# 원격 ENS 디렉토리의 파일 수를 반환하는 함수
def count_files_in_ens(directory_name):
    command = [rclone_path, 'lsf', f'{remote_server}:{remote_path}/{directory_name}/upper']
    result = subprocess.run(command, capture_output=True, text=True)
    files = result.stdout.split()
    # print(files)
    return len(files)


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
        
# 특정 ENS 디렉토리를 로컬로 동기화하는 함수
def sync_ens_directory(directory_name):
    remote_dir = f'{remote_server}:{remote_path}/{directory_name}'
    local_dir_path = os.path.join(local_path, directory_name)
    local_dir = f"{local_server}:{local_dir_path}"
    
    # 로컬 디렉토리 확인 및 생성
    print(local_dir)
    ensure_directory_exists(local_dir_path)
    subprocess.run([rclone_path, 'move', remote_dir, local_dir])
    # print(f'Synced {directory_name}')

# ENS 디렉토리 리스트를 가져오는 함수
def fetch_ens_directories():
    command = [rclone_path, 'lsf', '--dirs-only', f'{remote_server}:{remote_path}']
    # print(command)
    result = subprocess.run(command, capture_output=True, text=True)
    directories = result.stdout.split()
    # print(directories)
    # print([dir for dir in directories])
    return [dir for dir in directories]  # ENS가 포함된 디렉토리만 필터링

# 메인 함수
def main():
    ens_directories = fetch_ens_directories()
    # print(ens_directories)
    for directory in ens_directories:
        # print(directory)
        file_count = count_files_in_ens(directory)
        if file_count == 29:  # 디렉토리 내 파일 수가 29개일 때만 동기화
            sync_ens_directory(directory)
        else:
            print(f'{directory} contains {file_count} files, not syncing')

if __name__ == "__main__":
    main()
