import torch
import cv2
import numpy as np
import ipdb

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
    
    if size == 784:
        dct_matrix = torch.load("/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/dct_matrix/784.pt")
    elif size ==28:
        dct_matrix = torch.load("/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/dct_matrix/28.pt")
    elif size ==32:
        dct_matrix = torch.load("/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/dct_matrix/32.pt")
    else:
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

def batch_dct(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    m = input_tensor.shape[0]
    d = input_tensor.shape[1]

    dct_matrix = dct_matrix.to(input_tensor.device).expand(m,-1,-1)
    dct_output = torch.bmm(dct_matrix.transpose(1, 2), input_tensor.view(m,d,1)).squeeze()
    
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

def dct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    dct_output = torch.mm(torch.mm(dct_matrix.transpose(0, 1), input_tensor),dct_matrix)

    return dct_output

def idct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    
    size = input_tensor.shape[3]
    idct_matrix = torch.inverse(getDCTmatrix(size)).to(input_tensor.device)
    idct_output = torch.mm(torch.mm(idct_matrix.transpose(0, 1), input_tensor.squeeze()),idct_matrix)

    return idct_output

def batch_dct2(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    batch_size = input_tensor.shape[0]
    d = input_tensor.shape[2]

    dct_matrix = dct_matrix.to(input_tensor.device).expand(batch_size,-1,-1)
    dct2_output = torch.bmm(torch.bmm(dct_matrix.transpose(1, 2), input_tensor.view(batch_size,d,d)), dct_matrix)
    
    return dct2_output

def batch_idct2(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    batch_size = input_tensor.shape[0]
    d = input_tensor.shape[2]

    idct_matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(batch_size,-1,-1)
    idct2_output = torch.bmm(torch.bmm(idct_matrix.transpose(1, 2), input_tensor.view(batch_size,d,d)), idct_matrix)
    
    return idct2_output

