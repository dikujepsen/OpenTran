#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Non-interactive command line front end to NBody implementation"""

import time
from nbodyphysics_original import move, random_galaxy


def nbody_benchmark(bodies_list, time_step):
    """Run benchmark simulation without visualization"""

    x_max = 500
    y_max = 500
    z_max = 500
    dt = 1.0  # One year timesteps for better accuracy

    for bodies in bodies_list:
        galaxy = random_galaxy(x_max, y_max, z_max, bodies)

        start = time.time()
        for _ in range(time_step):
            move(galaxy, dt)
        stop = time.time()

        print 'Simulated ' + str(bodies) + ' bodies for ' \
            + str(time_step) + ' timesteps in ' + str(stop - start) \
            + ' seconds'
        print galaxy

if __name__ == '__main__':
    nbody_benchmark([8], 10)
