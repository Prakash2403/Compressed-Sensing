# Compressed-Sensing
A simple implementation of compressed sensing in python

------

### Original image

![](https://github.com/Prakash2403/Compressed-Sensing/blob/master/images/black-white-animal.jpg?raw=true)
------

### Resized image (Resized to 64 x 64)

![](https://github.com/Prakash2403/Compressed-Sensing/blob/master/images/Screenshot%202018-11-19%20at%2012.48.25%20AM.png?raw=true)

------

### Visually checking signal compressibility

![](https://github.com/Prakash2403/Compressed-Sensing/blob/master/images/Screenshot%202018-11-19%20at%2012.48.41%20AM.png?raw=true)
------

### Reconstructed image obtained after taking top 20% wavelet coefficients.

![](https://github.com/Prakash2403/Compressed-Sensing/blob/master/images/Screenshot%202018-11-19%20at%2012.48.52%20AM.png?raw=true)

------

### Reconstructed image obtained after L1 optimization. 

Note: In this case, measurement is obtained by multiplying measurement matrix with reconstructed image obtained **after** applying thresholding

![](https://github.com/Prakash2403/Compressed-Sensing/blob/master/images/Screenshot%202018-11-18%20at%205.42.42%20PM.png?raw=true)

------

### Reconstructed image obtained after L1 optimization.

Note: In this case, measurement is obtained by multiplying measurement matrix with original resized image.

![](https://github.com/Prakash2403/Compressed-Sensing/blob/master/images/Figure_1.png?raw=true)
