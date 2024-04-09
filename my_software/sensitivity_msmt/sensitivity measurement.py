from zhinst.toolkit import Session
import matplotlib.pyplot as plt
import time
import matplotlib
import pandas as pd
import pickle

matplotlib.use("Qt5Agg")

# IP address of the host computer where the Data Servers run
server_host = '192.168.113.190'
# A session opened to LabOne Data Server
session = Session(server_host)
# Open a session
hf2_session = Session(server_host)
# connect to the device
device = session.connect_device("DEV7279")


# Parameter
filename_save = 'test.pkl'
n_time_traces = 3
TOTAL_DURATION = 1 * n_time_traces # [s]
SAMPLING_RATE = 20e3 # [Hz]
N_SAMPLES = SAMPLING_RATE * n_time_traces # Number of points

SAMPLE_NODES = [
    device.demods[0].sample.x,
    device.demods[0].sample.y,
    #device.demods[0].sample.auxin0 # referernce signal
]

# Configure the module
daq_module = session.modules.daq
daq_module.device(device)
# daq_module.
# daq_module.setInt('/dev7279/demods/0/sinc', 1)                  # enable sinc filter
# daq_module.setInt('/dev7279/demods/0/order', 8)                 # filter order
# daq_module.setDouble('/dev7279/demods/0/timeconstant', 9.57619476e-05) # set filter bandwidth to 500 Hz

daq_module.type(0)                  # continuous acquisition
daq_module.grid.mode(2)             # how the acquired data is sample onto the matrix's horizontal axis - 2: linear interpolation
daq_module.count(1)                 # number of trigger events to acquire in single-shot mode
daq_module.duration(TOTAL_DURATION)
daq_module.grid.cols(N_SAMPLES)     # Specify the number of columns/samples in the returned data grid (matrix)

# Subscribe to each node
for node in SAMPLE_NODES:
    daq_module.subscribe(node)

# start and wait for the module to finish
daq_module.execute()
while not daq_module.raw_module.finished():
    time.sleep(1)

# Read results
result = daq_module.read(raw=False, clk_rate=device.clockbase())

# save results to pickle file
with open(filename_save, 'wb') as f:
    pickle.dump(result, f)

# data = pd.DataFrame({ "Time(s)": result[device.demods[0].sample.x][0].time,
#                             "X-Signal (V)": result[device.demods[0].sample.x][0].value[0]},
#                             columns=["Time(s)", "X-Signal (V)"])
# print(data)

# Plot the results
_, axis = plt.subplots(1, 1)
for sample_node in SAMPLE_NODES:
    axis.plot(
        result[sample_node][0].time,
        result[sample_node][0].value[0],
        label=sample_node
    )

axis.grid(True)
axis.legend()
axis.set_title("Data acquired through the DAQ module")
axis.set_xlabel("Time(s)")
axis.set_ylabel("Signal(V)")
plt.show()
