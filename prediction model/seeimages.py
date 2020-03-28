import random
import os
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

src = ''

# Check if the dataset has been downloaded. If not, direct user to download the dataset first
if not os.path.isdir(src):
    print("Dataset not found.")
    quit()
