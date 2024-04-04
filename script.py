# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:17:00 2024

@author: Sten
"""

# install_packages.py

import subprocess

packages_to_install = [
    "sounddevice",
    "numpy",
    "matplotlib",
    "Flask",
    "seaborn", 
    "pandas"
    "openpyxls"
    # Add more package names as needed
]

for package in packages_to_install:
    subprocess.call(["pip", "install", package])
print('Alles is gedownload')
