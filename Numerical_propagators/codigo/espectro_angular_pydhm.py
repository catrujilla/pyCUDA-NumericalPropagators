

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
import numpy as np
from timeit import default_timer as timer
# Angular spectrum
print ("Angular spectrum example")
# Load the input plane

#inp = utilities.imageRead('data/numericalPropagation samples/UofM-1-inv.jpg')
inp = utilities.imageRead('Numerical_propagators/imagenes/+3cm.tif')
utilities.imageShow(inp, 'input field')
start = timer()
# Numerical propagation using the angular spectrum
output = numericalPropagation.angularSpectrum(inp, 70000, 0.633, 6.9, 6.9)

# Display the output field
intensity = utilities.intensity(output, False)
print("without GPU:", timer()-start)	
utilities.imageShow(intensity, 'focused output field')