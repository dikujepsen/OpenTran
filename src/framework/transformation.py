import copy
import os
from visitor import *
from stringstream import *


class Transformation():
    """ Apply transformations to the original AST. Includes:
    1. Local Memory
    2. Stencil Local Memory
    3. Placing data in registers
    4. Transposing arrays
    5. Unrolling loops
    6. Adding defines
    7. Setting the number of dimensions to parallelize
    8. Setting the local work-group size
    9. Setting if we should read data back from the GPU

