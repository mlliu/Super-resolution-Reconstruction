#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
from PIL import Image
import logging.config
# Logging setup
#logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=False)
log = logging.getLogger(__name__)
from matplotlib import pyplot as plt
from PIL import Image


# %%


# Attempting to use mkl_fft (faster FFT library for Intel CPUs). Fallback is np
try:
    import mkl_fft as m

    fft2 = m.fft2
    ifft2 = m.ifft2
except (ModuleNotFoundError, ImportError):
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
finally:
    fftshift = np.fft.fftshift
    ifftshift = np.fft.ifftshift


# %%


def open_file(path: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Tries to load image data into a NumPy ndarray
    The function first tries to use the PIL Image library to identify and load
    the image. PIL will convert the image to 8-bit pixels, black and white.
    If PIL fails pydicom is the next choice.
    Parameters:
        path (str): The image file location
        dtype (np.dtype): image array dtype (e.g. np.float64)
    Returns:
        np.ndarray: a floating point NumPy ndarray of the specified dtype
    """

    log.info(f'Opening file: {path}')
    with Image.open(path) as f:
        img_file = f.convert('F')  # 'F' mode: 32-bit floating point pixels
        img_pixel_array = np.array(img_file).astype(dtype)
    log.info(f"Image loaded. Image size: {img_pixel_array.shape}")
    return img_pixel_array


# %%


def np_fft(img: np.ndarray):
        """ Performs FFT function (image to kspace)
        Performs FFT function, FFT shift and stores the unmodified kspace data
        in a variable and also saves one copy for display and edit purposes.
        Parameters:
            img (np.ndarray): The NumPy ndarray to be transformed
            out (np.ndarray): Array to store output (must be same shape as img)
        """
        kspacedata = np.zeros_like(img, dtype=np.complex64)
        return np.fft.fftshift(fft2(ifftshift(img))) 
    
def np_ifft(kspace: np.ndarray):
        """Performs inverse FFT function (kspace to [magnitude] image)
        Performs iFFT on the input data and updates the display variables for
        the image domain (magnitude) image and the kspace as well.
        Parameters:
            kspace (np.ndarray): Complex kspace ndarray
            out (np.ndarray): Array to store values
        """
        img = np.zeros_like(kspace, dtype=np.float32)
        return np.absolute(np.fft.fftshift(ifft2(ifftshift(kspace))))


def low_pass_filter(kspace: np.ndarray, factor: float):
    """Low pass filter removes the high spatial frequencies from k-space
    This function only keeps the center of kspace by removing values
    outside a circle of given size. The circle's radius is determined by
    the 'radius' float variable (0.0 - 100) as ratio of the lenght of
    the image diagonally
    Parameters:
        kspace (np.ndarray): Complex kspace data
        radius (float): Relative size of the kspace mask circle (percent)
    """
    kspace_data = np.zeros_like(kspace, dtype=np.complex64)
    kspace_data[:] = kspace[:]
    #original r
    #r = np.hypot(*kspace_data.shape) / 2
    #new r
    #new_r = np.sqrt(r**2 *(1/factor))
    #print("old radius and new radius:",r,new_r)
    rows, cols = np.array(kspace_data.shape, dtype=int)
    area_trans = rows*cols/factor
    r_trans = np.sqrt(area_trans/(2*np.pi))
    a, b = np.floor(np.array((rows, cols)) / 2).astype(int)
    y, x = np.ogrid[-a:rows - a, -b:cols - b]
    mask = x * x + y * y <= r_trans * r_trans*2
    
    kspace_data[~mask] = 0
    return kspace_data


    

def normalise(f: np.ndarray):
    """ Normalises array by "streching" all values to be between 0-255.
    Parameters:
        f (np.ndarray): input array
    """
    fmin = float(np.min(f))
    fmax = float(np.max(f))
    if fmax != fmin:
        coeff = fmax - fmin
        f[:] = np.floor((f[:] - fmin) / coeff * 255.)
        
    return f

def display(kspacedata):
    kspace_abs = np.absolute(kspacedata)
    kscale = 2
    if np.any(kspace_abs):
        scaling_c = np.power(10., kscale)
        kspace_abs = np.log1p(kspace_abs * scaling_c)
        ksapce_abs = normalise(kspace_abs)

    # 3. Obtain uint8 type arrays for QML display
    #self.image_display_data[:] = np.require(self.img, np.uint8)
    ksapce_abs= np.require(kspace_abs, np.uint8)

    return ksapce_abs


def generate(img,factors):
    kspace = np_fft(img)
    
    #ksapce low_pass_filter
    img_trans = np.zeros((len(factors)+1,img.shape[0],img.shape[1]),dtype=np.float32)
    kspace_trans = np.zeros((len(factors)+1,img.shape[0],img.shape[1]),dtype=np.complex64)

    kspace_trans[0,:] = kspace
    img_trans[0,:] = img
    for i in range(len(factors)):
        factor = factors[i]
        kspace_trans[i+1,:] = low_pass_filter(kspace,factor =factor)

        #kspace to img
        img_trans[i+1,:] = np_ifft(kspace_trans[i+1,:])
        
    return kspace_trans,img_trans


# %%
