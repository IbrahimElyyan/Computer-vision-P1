import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)

    ### YOUR CODE HERE

    # Smooth image with Gaussian kernel
    kernel = gaussian_kernel(5,1)
    img_smooth = filter2d(img, kernel)
    
    # Compute x and y derivate on smoothed image
    x_der = partial_x(img_smooth)
    y_der = partial_y(img_smooth)

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(x_der**2 + y_der**2)
    
    # Visualize results
    plt.figure(figsize=(20,20))
    
    
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()

