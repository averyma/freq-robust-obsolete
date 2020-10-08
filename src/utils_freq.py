import torch
import cv2
import numpy as np

def rgb2gray(rgb_input):
    """
        reference: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """
    # batch operation
    if len(rgb_input.shape) == 4:
        gray_output = rgb_input[:, 0, :, :]*0.299 + \
                        rgb_input[:, 1, :, :]*0.587 + \
                        rgb_input[:, 2, :, :]*0.114
    # single image operation
    elif len(rgb_input.shape) == 3:
        gray_output = rgb_input[0, :, :]*0.299 + \
                        rgb_input[1, :, :]*0.587 + \
                        rgb_input[2, :, :]*0.114
 
    else:
        raise  NotImplementedError("Input dimension not supported. Check tensor shape!")
 
    return gray_output

def getDCTmatrix(device):
    """
        Computed using C_{jk}^{N} found in the following link:
        https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        Verified with cv2.dct(), error less than 1.1260e-06.
    """
    dct_matrix = torch.zeros([32, 32], device=device)

    for i in range(0, 32):
        for j in range(0, 32):
            if j == 0:
                dct_matrix[i, j] = np.sqrt(1/32)*np.cos(np.pi*(2*i+1)*j/2/32)
            else:
                dct_matrix[i, j] = np.sqrt(2/32)*np.cos(np.pi*(2*i+1)*j/2/32)

    return dct_matrix

def dct(input_tensor, device):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """
    dct_matrix = getDCTmatrix(device).view(1, 32, 32).expand(input_tensor.shape[0], -1, -1)
    dct_output = torch.bmm(torch.bmm(dct_matrix.transpose(1, 2), input_tensor), dct_matrix)

    return dct_output
