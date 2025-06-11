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


def test_print():
    heading("wow")
    sub_heading("oolala")
    sub_heading("huhuhaha")
    heading("ahem")
    heading("wow")
    heading("ahem")
    sub_heading("oolala")
    sub_heading("huhuhaha")
    sub_heading("oolala")
    sub_heading("huhuhaha")
    heading("wow")
    heading("ahem")
    heading("wow")
    heading("ahem")


