"""
Title-->            Numerical propagation examples.
Author-->           Ana Doblas, Carlos Trujillo, Raul Castaneda,
Date-->             05/09/2022
                    University of Memphis
                    Optical Imaging Research lab (OIRL)
                    EAFIT University
                    Applied Optics Group
Abstract -->        Script that implements the different methods to propagate the resulting complex field data
Links-->          - https://github.com/catrujilla/pyDHM
"""

from pyDHM import utilities
from pyDHM import numericalPropagation
from timeit import default_timer as timer
start = timer()

print ("Fresnel transform example")

# Fresnel transform and speckle reduction via HM2F

# Load the input plane
inp = utilities.imageRead('Numerical_propagators/imagenes/horse.bmp')

# FT of the hologram
#ft_holo = utilities.FT(inp)
#ft_holo = utilities.intensity(ft_holo, True)
#utilities.imageShow(ft_holo, 'FT hologram')

# Spatial filter

# Numerical propagation using the Fresnel transforms
output = numericalPropagation.fresnel(inp, -450, 0.000633, 0.005000, 0.005000)

# Display the output field
amplitude = utilities.amplitude(output, True)

# HM2F to reduce the speckle
denoise = utilities.HM2F(amplitude, 7, False, False)

#amplitude = utilities.amplitude(denoise, False)
print("without GPU:", timer()-start)	
