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

inp = utilities.imageRead('Numerical_propagators/imagenes/die_1.jpg')
utilities.imageShow(inp, 'input field')
start = timer()
# Numerical propagation using the Fresnel transforms
output = numericalPropagation.bluestein(inp, 30/100, 632.8e-9, 7e-6, 7e-6, 4.5e-5, 4.5e-5)

# Display the output field
amplitude = utilities.amplitude(output, False)
print("without GPU:", timer()-start)	
utilities.imageShow(amplitude, 'output field')
