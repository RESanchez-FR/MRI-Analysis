# Intro to MRI analysis and Thesis Project In Development

Using Image Processing Technqiues from MRI Dicom Images, this project aims to do the Following.

1. Plot Visualizations on Strain Data and Diffusion Tensor Data
2. Perform Fiber Aligned Strain using both Data Sets
3. Aim to understand the correlation between Muscle Deformation and the Fibers during a contraciton Cycle improving our understanding in muscle movement

# Software Tools Used in this Process
1. Matlab
2. Python

# Folder Descriptions:

# Strain Visuals
1.  This section plots colormaps of the strain data sets used in this project. The data is 6-Dimensions and we focus on the Principal EigenVector with its componets (X,Y,Z)
2.  The Principal EigenVector of the Strain Data is arguably the direction that is aligned most with the muscle fibers and this is explored in great length
3.  We analyze this data set and plot the data in a RGB color map where we designate each direction of the Principal EigenVector to be Red = X , Green = Y , Blue = Z
4.  Utilizing this information we are able visualize the direction the muscles move during the contraction cycle
5.  A Gif File is Stored iterating through the different slices of the image acquisition process
