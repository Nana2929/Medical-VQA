import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from skimage.filters import median
from skimage.filters.rank import median as rank_median
from skimage import io


# Apply median filtering
def median_filter(img):
    return median(img)

# Apply Wiener filtering
def wiener_filter(img):
    return wiener(img)