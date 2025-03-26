import numpy as np


# def smooth_strain_data(strain_data, degree=3):
#     """
#     Smooths strain tensor data by fitting a polynomial of specified degree across the last dimension.

#     Parameters:
#     - strain_data: array_like, shape (x, y, z, t, component1, component2) - Strain tensor data.
#     - degree: int, optional - Degree of the polynomial. Default is 3.

#     Returns:
#     - smoothed_data: array_like - Smoothed strain tensor data.
#     """
#     # Initialize smoothed data array
#     smoothed_data = np.zeros_like(strain_data)
    
#     component1 = 0
#     # Iterate over all but the last dimension
#     for x in range(strain_data.shape[0]):
#         for y in range(strain_data.shape[1]):
#             for z in range(strain_data.shape[2]):
#                 for t in range(strain_data.shape[3]):
#                     # for component1 in range(strain_data.shape[4]):
#                         # Fit a polynomial to each component across the last dimension
#                     coefficients = np.polyfit(range(strain_data.shape[5]), strain_data[x, y, z, t, component1, :], degree)
#                     polynomial = np.poly1d(coefficients)
#                     smoothed_data[x, y, z, t, component1, :] = polynomial(range(strain_data.shape[5]))
    
#     return smoothed_data



def smooth_sixth_dimension(data, degree=3):
    """
    Smooths data along the 6th dimension by fitting a polynomial of specified degree.

    Parameters:
    - data: array_like, shape (x, y, z, t, component, elements) - Data to be smoothed.
    - degree: int, optional - Degree of the polynomial. Default is 3.

    Returns:
    - smoothed_data: array_like - Smoothed data.
    """
    # Initialize smoothed data array
    smoothed_data = np.zeros_like(data)
    
    # Iterate over all but the last dimension
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                for t in range(data.shape[3]):
                    for component in range(data.shape[4]):
                        # Fit a polynomial to the data along the 6th dimension
                        coefficients = np.polyfit(range(data.shape[5]), data[x, y, z, t, component, :], degree)
                        polynomial = np.poly1d(coefficients)
                        
                        # Generate smoothed data for the 6th dimension
                        smoothed_data[x, y, z, t, component, :] = polynomial(range(data.shape[5]))
    
    return smoothed_data
