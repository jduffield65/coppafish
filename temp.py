import time
from coppafish import run_pipeline

config_file = '/home/paul/Documents/coppafish/minnie/minnie.ini'

start_time = time.time()

run_pipeline(config_file, parallel=False)

end_time = time.time()

print(f'Time taken to run: {round(end_time-start_time, 1)}s')
