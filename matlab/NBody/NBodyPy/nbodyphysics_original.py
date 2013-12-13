#!/usr/bin/python
# -*- coding: utf-8 -*-

"""NBody in N^2 complexity
Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import numpy

# By using the solar-mass as the mass unit and years as the standard time-unit
# the gravitational constant becomes 1

G = 1.0


def calc_force(a, b, dt):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)/((x_b-x_a)/r)
    """

    r = ((b['x'] - a['x']) ** 2 + (b['y'] - a['y']) ** 2 + (b['z']
         - a['z']) ** 2) ** 0.5
    a['vx'] += G * a['m'] * b['m'] / r ** 2 * ((b['x'] - a['x']) / r) \
        / a['m'] * dt
    a['vy'] += G * a['m'] * b['m'] / r ** 2 * ((b['y'] - a['y']) / r) \
        / a['m'] * dt
    a['vz'] += G * a['m'] * b['m'] / r ** 2 * ((b['z'] - a['z']) / r) \
        / a['m'] * dt


def move(galaxy, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """

    for i in galaxy:
        for j in galaxy:
            if i != j:
                calc_force(i, j, dt)

    for i in galaxy:
        i['x'] += i['vx']
        i['y'] += i['vy']
        i['z'] += i['vz']


def random_galaxy(
    x_max,
    y_max,
    z_max,
    n,
    ):
    """Generate a galaxy of random bodies"""

    max_mass = 40.0  # Best guess of maximum known star

    # We let all bodies stand still initially

    return [{
        'm': numpy.random.random() * numpy.random.randint(1, max_mass)
            / (4 * numpy.pi ** 2),
        'x': numpy.random.randint(-x_max, x_max),
        'y': numpy.random.randint(-y_max, y_max),
        'z': numpy.random.randint(-z_max, z_max),
        'vx': 0,
        'vy': 0,
        'vz': 0,
        } for _ in xrange(n)]


