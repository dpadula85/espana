#!/usr/bin/env python

import setuptools
from numpy.distutils.core import Extension, setup

setup(
    name="ESPAna",
    version="1.0",
    author="Daniele Padula, Leonardo Barneschi",
    author_email="dpadula85@yahoo.it, leonardo.barneschi@student.unisi.it",
    description="A python package to compute coulomb electrostatic potentials",
    url="https://github.com/dpadula85/espana",
    packages=setuptools.find_packages(),
    ext_modules=[ Extension('espana.inter', ['espana/inter.f90'],
                  extra_f90_compile_args=['-fopenmp', '-lgomp'],
                  extra_link_args=['-lgomp']) ],
	scripts=['espana/bin/compute_esp'],
    entry_points={ 
        'console_scripts' : [
            'grid_gen=espana.grid_gen:main'
            ]
        },
    zip_safe=False
)
