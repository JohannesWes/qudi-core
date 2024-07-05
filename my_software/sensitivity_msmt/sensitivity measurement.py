import time
import pickle
import matplotlib.pyplot as plt
import matplotlib
from zhinst.toolkit import Session
from my_software.automation.qudi_remote_control import OdmrRemoteControl

matplotlib.use("Qt5Agg")

# Constants
SERVER_HOST = '192.168.113.190'
DEVICE_ID = "DEV7279"
FILENAME_SAVE = 'auswertung/cobolt_04-07-24-23Uhr18_0.02OPX_250kHzfdev_7kHzfmod/cobolt_04-07-24-23Uhr18_0.02OPX_250kHzfdev_7kHzfmod_LIA250Hz.pkl'
N_TIME_TRACES = 5
TOTAL_DURATION = 1 * N_TIME_TRACES  # [s]
SAMPLING_RATE = 20e3  # [Hz]
N_SAMPLES = int(SAMPLING_RATE * N_TIME_TRACES)  # Number of points



def data_acquisition():
    """Acquire demodulator data using the DAQ module of the ZI LIA."""

    # Create a session, connect to the device and the DAQ module
    session = Session(SERVER_HOST)
    device = session.connect_device(DEVICE_ID)
    daq_module = session.modules.daq

    daq_module.device(device)
    daq_module.type(0)  # Continuous acquisition
    daq_module.grid.mode(2)  # Linear interpolation
    daq_module.count(1)  # Number of trigger events to acquire in single-shot mode
    daq_module.duration(TOTAL_DURATION)
    daq_module.grid.cols(N_SAMPLES)  # Number of columns/samples in the returned data grid

    # subscribe to the demodulator sample nodes
    demod_sample_nodes = [
        device.demods[0].sample.x,
        device.demods[0].sample.y,
    ]
    for node in demod_sample_nodes:
        daq_module.subscribe(node)

    # Start the DAQ module and wait until the data acquisition finished
    daq_module.execute()
    while not daq_module.raw_module.finished():
        time.sleep(1)

    result = daq_module.read(raw=False, clk_rate=device.clockbase())

    times = result[demod_sample_nodes[0]][0].time
    x_value = result[demod_sample_nodes[0]][0].value[0]
    y_value = result[demod_sample_nodes[1]][0].value[0]

    return times, x_value, y_value


def save_results(filename, result):
    """Save results to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def plot_results(times, x, y):
    """Plot the results using matplotlib."""
    _, axis = plt.subplots(1, 1)

    for data in (x, y):
        axis.plot(
            times,
            data,
            label="x"
        )

    axis.grid(True)
    axis.legend()
    axis.set_title("Data acquired through the DAQ module")
    axis.set_xlabel("Time(s)")
    axis.set_ylabel("Signal(V)")
    plt.show()

def sensitivity_measurement():
    """Perform a sensitivity measurement."""

    times, x_value, y_value = data_acquisition()
    plot_results(times, x_value, y_value)
    # print("result",time)
    # print("demod_sample_nodes",demod_sample_nodes)


if __name__ == "__main__":
    sensitivity_measurement()