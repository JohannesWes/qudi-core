import numpy as np
from scipy.constants import mu_0
from RsNgx import *
import pyvisa as visa
from pathlib import Path
import importlib.util
import sys
import pickle

path = Path("").absolute()
path_to_mod = str(path.parents[0] / "johannes_tools" / "data_analysis" / "fitting.py")
spec = importlib.util.spec_from_file_location("fitting", path_to_mod)
johannes_tools = importlib.util.module_from_spec(spec)
sys.modules["fitting"] = johannes_tools
spec.loader.exec_module(johannes_tools)
from fitting import evaluate_hyperfine # noqa


# -------------------------------------------------------------------------------------------------------------------- #


from NGP_control import *

if __name__ == '__main__':
    ngp = NGP_instance()

    ngp.set_channel(3, 0, 0.15)

    ngp.output_on()
    ngp.deactivate_channel(1)
    ngp.deactivate_channel(2)
    ngp.deactivate_channel(3)
    ngp.output_off()
    ngp.driver.close()
    print('NGP closed')