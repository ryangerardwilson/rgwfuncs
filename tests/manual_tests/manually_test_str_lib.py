import sys
import os
import math
import csv
import tempfile
import pandas as pd
import numpy as np

# flake8: noqa: E402
# Allow the following import statement to be AFTER sys.path modifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rgwfuncs.str_lib import title, heading, sub_heading
from manually_test_str_lib_as_module import test_print

title("Hello")
heading("wow")
sub_heading("oolala")
sub_heading("huhuhaha")
heading("ahem")
heading("wow")
heading("ahem")
sub_heading("oolala")
sub_heading("huhuhaha")
print("dsjdsakdjsakjdkasj")
sub_heading("oolala")
sub_heading("huhuhaha")
heading("wow")
heading("ahem")
test_print()
heading("wow")
heading("ahem")

"""
[1/8] WOW
[1/8] -> (1/2) oolala
[1/8] -> (2/2) huhuhaha
[2/8] AHEM
[3/8] WOW
[4/8] AHEM
[5/8] WOW
[6/8] AHEM
[7/8] WOW
[8/8] AHEM

"""




