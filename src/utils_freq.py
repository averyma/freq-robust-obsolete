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

def getDCTmatrix(size):
    """
        Computed using C_{jk}^{N} found in the following link:
        https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        Verified with cv2.dct(), error less than 1.1260e-06.
        
        output: DCT matrix with shape (size,size)
    """
    dct_matrix = torch.zeros([size, size])

    for i in range(0, size):
        for j in range(0, size):
            if j == 0:
                dct_matrix[i, j] = np.sqrt(1/size)*np.cos(np.pi*(2*i+1)*j/2/size)
            else:
                dct_matrix[i, j] = np.sqrt(2/size)*np.cos(np.pi*(2*i+1)*j/2/size)

    return dct_matrix

def dct(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    dct_output = torch.mm(dct_matrix.transpose(0, 1), input_tensor)

    return dct_output

def idct(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    idct_matrix = torch.inverse(dct_matrix)
    idct_output = torch.mm(idct_matrix.transpose(0, 1), input_tensor)

    return idct_output

def dct2(input_tensor, device):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size, device)
    dct_output = torch.mm(torch.mm(dct_matrix.transpose(0, 1), input_tensor),dct_matrix)

    return dct_output

def idct2(input_tensor, device):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    
    size = input_tensor.shape[0]
    idct_matrix = torch.inverse(getDCTmatrix(size, device))
    idct_output = torch.mm(torch.mm(idct_matrix.transpose(0, 1), input_tensor),idct_matrix)

    return idct_output

def batch_dct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on the entire batch
        input shape: (batch, size, size)
        output shape: (batch, size, size)
    """ 
    
    size = input_tensor.shape[1]
    dct_matrix = getDCTmatrix(size, device).view(1, size, size).expand(input_tensor.shape[0], -1, -1)
    dct_output = torch.bmm(torch.bmm(dct_matrix.transpose(1, 2), input_tensor), dct_matrix)

    return dct_output

