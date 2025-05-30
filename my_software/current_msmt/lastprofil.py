import time
from my_software.automation.power_supply_NGP_control import NGP_instance
import numpy as np

# Setting up the R&S NGP power supply
ngp_active_channel = 1
ngp = NGP_instance(usb_address="USB0::0x0AAD::0x0197::5601.4007k03-101169::INSTR")
ngp.activate_channel(ngp_active_channel)
ngp.output_on()

# Current values to cycle through
current_array = [0.001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, .175, 0.2, 0.225, 0.25, 0.225, 0.2, 0.175,  0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.001] # A
current_array2 = [0.001, 0.5, 0.001, 0.5, 0.001, 0.5, 0.001]

# define a sine wave array
current_array = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.005 + 0.006
# now a square wave instead
current_array2 = np.sign(np.sin(np.linspace(0, 2 * np.pi, 500))) * 0.005 + 0.006

ngp.set_voltage(ngp_active_channel, 60)  # Keeping the same voltage setting
ngp.set_current(ngp_active_channel, 0.001)
# Set each current value with a 5-second delay
try:
    print("Starting current cycling sequence")
    while True:  # Infinite loop - can be stopped with Ctrl+C
        # for current in current_array:
        #     print(f"Setting current to: {current} A")
        #     ngp.set_current(ngp_active_channel, current)
        #     time.sleep(0.001)
        # for current in current_array:
        #     print(f"Setting current to: {current} A")
        #     ngp.set_current(ngp_active_channel, current)
        #     time.sleep(0.001)
        # for current in current_array:
        #     print(f"Setting current to: {current} A")
        #     ngp.set_current(ngp_active_channel, current)
        #     time.sleep(0.001)
        # for current in current_array2:
        #     print(f"Setting current to: {current} A")
        #     ngp.set_current(ngp_active_channel, current)
        #     time.sleep(0.001)
        #     #time.sleep(10)  # Wait 5 seconds before changing to next current value
        for current in current_array2:
            print(f"Setting current to: {current} A")
            ngp.set_current(ngp_active_channel, 0.01)
            time.sleep(10)  # Wait 5 seconds before changing to next current value
except KeyboardInterrupt:
    print("Current cycling interrupted by user")
finally:
    ngp.close()
    print('NGP closed')