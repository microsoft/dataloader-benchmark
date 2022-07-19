import time

import mlflow
import psutil

now = 0
dt_sleep = 1
while True:
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent()
    swap = psutil.swap_memory().percent
    mlflow.log_metric(key="monitor/ram", value=ram, step=now)
    mlflow.log_metric(key="monitor/cpu", value=cpu, step=now)
    mlflow.log_metric(key="monitor/swap", value=swap, step=now)
    time.sleep(dt_sleep)
    now += dt_sleep
