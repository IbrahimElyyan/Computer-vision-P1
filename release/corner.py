import numpy as np
from utils import gaussian_kernel, filter2d, partial_x, partial_y
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    response = None
    
    ### YOUR CODE HERE
    # Computing the x and y derivatives
    x_der = partial_x(img)
    y_der = partial_y(img)
    
    # Compute product of derivatives
    xx_der = x_der**2
    yy_der = y_der**2
    xy_der = x_der*y_der
    
    # Compute sum of products
    sum_xx = filter2d(xx_der, gaussian_kernel(window_size))
    sum_yy = filter2d(yy_der,gaussian_kernel(window_size))
    sum_xy = filter2d(xy_der,gaussian_kernel(window_size))
    
    # Compute reponse using formula
    det = xx_der*yy_der - xy_der**2
    trace = xx_der + yy_der
    response = det - k(trace**2)
    
    ### END YOUR CODE

    return response

def main():
    img = imread('release\building.jpg', as_gray=True)

    ### YOUR CODE HERE
    
    # Compute Harris corner response
    response = harris_corners(img)
    
    # Threshold on response, I chose 0.01
    thresh = 0.01
    response[response<thresh] = 0
    
    # Perform non-max suppression by finding peak local maximum
    coordinates = peak_local_max(response, min_distance=20)
    
    # Visualize results
    
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()
