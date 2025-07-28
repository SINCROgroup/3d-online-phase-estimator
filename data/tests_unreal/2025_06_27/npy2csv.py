import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt


npy_path  = "ex02_positions_history"
time_path = "ex02_time_history"

# Load numpy arrays
arr  = np.load(npy_path + ".npy")
time = np.load(time_path + ".npy")
# Ensure arr is at least 2D for CSV writing
if arr.ndim == 1:
	arr = arr.reshape(-1, 1)
elif arr.ndim == 3:
	arr = arr.reshape(arr.shape[0], -1)
# Ensure time is a column vector
time = time.reshape(-1, 1)
# Check matching lengths
time = time[:-1]
if time.shape[0] != arr.shape[0]:
	print("Error: time and data arrays have different lengths.")
	sys.exit(1)
# Concatenate time as first column
arr_with_time = np.hstack((time, arr))
# Prepare CSV path
base, _ = os.path.splitext(npy_path)
csv_path = base + '.csv'
# Write to CSV with header
header = ['time'] + [f'col{i+1}' for i in range(arr.shape[1])]
with open(csv_path, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(header)
	writer.writerows(arr_with_time)
print(f"Saved CSV to {csv_path}")
