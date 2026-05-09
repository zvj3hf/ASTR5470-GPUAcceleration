"""
Configuring CUDA and GPU launch for a 1D and 2D problem.
"""

import math

def configure_launch_1d(n, threads_per_block=256):
"""
Figure out launch configuration for a 1D problem. 

Parameters:

n: int
    number of elements to compute
threads_per_block: int
    number of threads per block on CUDA

Returns:

blocks_per_grid: int
    number of blocks needed based on dataset
threads_per_block: int
    number of threads per block

"""

#use math.ceil to round up, prefer to overestimate blocks needed

blocks_per_grid = math.ceil(n/threads_per_block)
return blocks_per_grid, threads_per_grid

def configure_launch_2d(shape, threads_per_block=(16,16)):
"""
Figure out launch configuration for a 2D problem.

Parameters:

shape: tuple
    shape of the 2D grid
threads_per_block: tuple
    threads per block in x,y

Returns:

blocks_per_grid: tuple
    blocks needed in x,y
threads_per_block: tuple
    threads per block in x,y

"""

nx,ny = shape

blocks_per_grid = (math.ceil(nx/threadsperblock[0]), math.ceil(ny/threads_per_block[1]))

return blocks_per_grid, threads_per_block