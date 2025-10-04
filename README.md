Conv2D -> creates grids of number, higher values = stronger activations (detection of edges/texture, etc.)
    - in_channels = number of channels in the input image (1 - Greyscale, 3 - RGB)
    - out_channels = number of filters (feature maps) the layer learns
    - kernel_size = size of the conv filter (n x n) or tuple: (h x w)
    - padding = adds pixels around the input

MaxPool2d -> downsamples feature maps by sliding kernel (n x n) over them and for each window replacing w/ single maximum value
    - kernel_size = size of the window over what the max is taken

EXAMPLE:

Feature Map:
    1  3  2  4
    5  6  1  2
    7  2  8  3
    4  9  2  1

MaxPool2d(2) -> 2x2 kernels 
   
    | 1  3 | 2  4
    | 5  6 | 1  2
      7  2   8  3
      4  9   2  1

RESULTS:
    6  4
    9  8
      
