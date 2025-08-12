import sys
import re
import csv

'''epoch time: 188.30 sec, parameter memory: 41.42 GB, memory: 47.280882176 GB'''

def write_csv_header(csv_file):
    with open(csv_file, 'w') as f_csv:
        csv_writer = csv.writer(f_csv)
        header = ['model', 'performance', 'metric']
        csv_writer.writerow(header)

def write_csv(csv_file, model, th):
    with open(csv_file, 'a') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(['hy_video_'+model, th, 'seconds'])

def parse_log(log_file, model, csv_file):
    with open(log_file, 'r') as f_log:
        for line in f_log.readlines():
            if 'epoch time:' in line:
                match = re.search(r"epoch time: (\d+\.\d+)", line)
                if match:
                    th = match.group(1)
                    #print(f'performance: {th} samples/s')
                    write_csv(csv_file, model, th)
                else:
                    print(f'ERROR: Incorrect line format.')

write_csv_header(sys.argv[2])
parse_log(sys.argv[1], 'inference', sys.argv[2])
