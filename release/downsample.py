import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d

def main():
    
    # load the image
    im = imread('release\\paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        #subsample image 
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i+1)
        plt.imshow(im_subsample)
        plt.axis('off')

    # subsampling without aliasing, visualize results on 2nd row

    #### YOUR CODE HERE
    im_subsample = im.copy()
    for i in range(N_levels):
        # Apply a Gaussian filter for anti-aliasing
        kernel = gaussian_kernel(5, 1)
        
        # Separate color channels
        r, g, b = im_subsample[:, :, 0], im_subsample[:, :, 1], im_subsample[:, :, 2]
        
        # Apply Gaussian filter and subsample each color channel
        r_subsample = filter2d(r, kernel)[::2, ::2]
        g_subsample = filter2d(g, kernel)[::2, ::2]
        b_subsample = filter2d(b, kernel)[::2, ::2]
        
        # Combine color channels back into a single image
        im_subsample = np.stack([r_subsample, g_subsample, b_subsample], axis=2)
        
        plt.subplot(2, N_levels, N_levels+i+1)
        plt.imshow(im_subsample)
        plt.axis('off')
    
    plt.show()
    #### END YOUR CODE
    
if __name__ == "__main__":
    main()


