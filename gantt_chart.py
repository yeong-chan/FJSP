import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import tkinter as tk
from tkinter import ttk

# Global variables
vline = None
x_coordinate_label = None


def on_mouse_motion(event):
    global vline
    x, y = event.xdata, event.ydata
    if x is not None:
        x_discrete = round(x)
        if vline is not None:
            vline.remove()
        last_finish_time = job_info['Time_y'].max()  # 마지막 작업의 종료 시간
        if x_discrete < last_finish_time:  # 커서 위치가 종료 시간 이후
            vline = ax.axvline(x_discrete, color='red')
            current_jobs = find_current_jobs(x_discrete)
            job_info_label.config(text=f'현재 작업: {", ".join(current_jobs)}')
        else:  # 커서 위치가 종료 시간 이전
            x_discrete = last_finish_time
            vline = ax.axvline(x_discrete, color='red')
            job_info_label.config(text=f'Finish, Total tardiness: {total_tardiness:.2f}')
        canvas.draw()
        x_coordinate_label.config(text=f'current time : {x_discrete:d}')  # discrete x 좌표를 표시 (빨간선)


def on_closing():
    root.quit()


# tkinter 창 생성
root = tk.Tk()
root.title("마우스 위치에 이산적인 빨간 세로줄 및 X 좌표를 표시합니다.")
root.protocol("WM_DELETE_WINDOW", on_closing)

colorlist = [  # 70
    '#E6194B', '#3CB44B', '#FFE119', '#0082C8', '#F58231',
    '#911EB4', '#46F0F0', '#F032E6', '#D2F53C', '#FABEBE',
    '#008080', '#A9A9A9', '#800000', '#808000', '#000080',
    '#F0E68C', '#CD5C5C', '#F5DEB3', '#4682B4', '#F4A460',
    '#E6BEFF', '#9A6324', '#FFFAC8', '#800000', '#AA6E28',
    '#808000', '#007FFF', '#1CE6FF', '#614051', '#E8E474',
    '#8B4513', '#7B68EE', '#FF4500', '#2E8B57', '#DA70D6',
    '#5F9EA0', '#D2691E', '#8B008B', '#6495ED', '#FFF8DC',
    '#DC143C', '#00FFFF', '#008B8B', '#B8860B', '#A9A9A9',
    '#006400', '#BDB76B', '#8B4513', '#FF8C00', '#9932CC',
    '#8B0000', '#E9967A', '#8A2BE2', '#A52A2A', '#D2691E',
    '#FF7F50', '#6A5ACD', '#4B0082', '#00FF7F', '#9370DB',
    '#3CB371', '#FA8072', '#9400D3', '#FF1493', '#B0C4DE',
    '#1E90FF', '#40E0D0', '#D2B48C', '#228B22', '#D8BFD8',
    '#191970', '#8B008B', '#00CED1', '#DDA0DD', '#B0E0E6'
]

current_directory = os.getcwd()
print("현재 작업 디렉토리:", current_directory)
# CSV 파일을 읽어옵니다. 파일 경로를 적절하게 변경하세요.
file_path = current_directory + "\\environment\\result\\random\\log_1.csv"
df = pd.read_csv(file_path)

# 작업 시작 및 작업 완료 이벤트를 필터링하여 추출합니다.
start_events = df[df['Event'] == 'Work Start']
finish_events = df[df['Event'] == 'Work Finish']
total_tardiness = df['Tardiness'].dropna().sum()

# 작업 시작 이벤트와 작업 완료 이벤트를 병합하여 작업 정보를 추출합니다.
job_info = pd.merge(start_events, finish_events, on='Job')

# Gantt 차트를 그리기 위한 데이터를 구성합니다.
fig, ax = plt.subplots(figsize=(20, 8))

# pattern = r'Op (\d+)-\d+'

# machine 값을 추출하고 숫자로 변환
job_info['Machine'] = job_info['Machine_x'].str.extract(r'(\d+)').astype(int)

# 머신(Machine) 인덱스를 기준으로 오름차순으로 정렬
job_info = job_info.sort_values(by='Machine')

# 각 작업의 작업 시작과 작업 완료 시간을 이용하여 직사각형을 그립니다.
barh_data = []
for index, row in job_info.iterrows():
    start_time = row['Time_x']
    end_time = row['Time_y']
    machine = row['Machine_x']
    job = row['Job']
    if job[-2] == "-":
        if job[-4] == " ":
            idx = job[-3]
        else:
            idx = job[-4:-2]
    else:  # job [-3] == "-"
        if job[-5] == " ":
            idx = job[-4]
        else:
            idx = job[-5:-3]
    barh_data.append((machine[-1], end_time - start_time, start_time, int(idx)))

    # ax.barh(machine, end_time - start_time, left=start_time, label=f'Job {idx}', alpha=0.6, color=colorlist[int(idx)])

barh_data = sorted(barh_data, key=lambda x: x[0])   # 그릴 순서대로 정렬

for data in barh_data:
    machine, width, left, job_number = data
    ax.barh(machine, width, left=left, label=f'Job {job_number}', alpha=0.6, color=colorlist[job_number])

# 그래프를 tkinter 창에 삽입하기
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
# 캔버스 배치 설정 (원하는 위치 및 크기 조정)
canvas_widget.grid(row=0, column=0)
# 마우스 이벤트를 처리하기 위해 콜백 함수 등록
fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
# X 좌표를 표시할 레이블 추가
x_coordinate_label = ttk.Label(root, text="current time: ")
x_coordinate_label.grid(row=1, column=0, padx=10, pady=10)

"""
현재 마우스 커서 위치에 일하고 있는 job info
"""

def find_current_jobs(current_time):
    current_jobs = []
    for index, row in job_info.iterrows():
        start_time = row['Time_x']
        end_time = row['Time_y']
        machine = row['Machine_x']
        job = row['Job']
        if start_time <= current_time < end_time:
            current_jobs.append(f'{machine} : {job}')
    # current_jobs = sorted(current_jobs, key=lambda x: [int(d) for d in re.findall(r'\d+', x)])
    return current_jobs


# 마우스 이벤트를 처리하기 위한 콜백 함수 등록
fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

# 현재 작업 정보를 표시할 레이블 추가
job_info_label = ttk.Label(root, text="현재 작업: ")
job_info_label.grid(row=2, column=0, padx=10, pady=10)
"""
"""

# 그래프 스타일 및 레이블을 설정합니다.
ax.set_xlabel('Time')
ax.set_ylabel('Machine')
ax.set_title('Gantt Chart')
ax.invert_yaxis()  # 머신을 위에서 아래로 표시하도록 역순으로 정렬합니다.

# ax.set_xlim(0, 1100)

# 범례를 추가합니다.
# 중복된 라벨을 제거한 후 legend 표시
unique_labels = []
unique_handles = []
for handle, label in zip(*ax.get_legend_handles_labels()):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)
ax.legend(unique_handles, unique_labels, loc='upper right')

root.mainloop()
# Gantt 차트를 표시합니다.
# plt.show()
