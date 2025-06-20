#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:11:25 2024

@author: Javiera Jilberto Vallejos
"""
import os
import numpy as np
from fibergen.FibGen import FibGen
import cheartio as chio

mesh_folder = 'test_data/mesh/'
data_path = 'test_data/data/'
mesh_path = mesh_folder + 'bv_model'

if not os.path.exists(data_path): os.mkdir(data_path)

# load boundaries and apex
boundaries = chio.pfile_to_dict(mesh_folder + 'boundaries.P', fmt=int)
apex_id = boundaries['apex']  # lv apex point in the epicardium

# Constants (in deg)
params = {
    # A = alpha angle
    'AENDORV' : 90,
    'AEPIRV' : -25,
    'AENDOLV' : 60,
    'AEPILV' : -60,

    # AOTENDOLV = 90
    'AOTENDOLV' : 0, # Changed 24/04/2020
    'AOTENDORV' : 90,
    'AOTEPILV' : 0,
    'AOTEPIRV' : 0,
    'ATRIENDO' : 0, # Changed 13/06/2020

    # B = beta angle
    'BENDORV' : 0,
    'BEPIRV' : 0,
    'BENDOLV' : 0,
    'BEPILV' : 0,
}

# Reading mesh
mesh = chio.read_mesh(mesh_path, meshio=True)
bdata = chio.read_bfile(mesh_path)

# Generating fibers
fibgen = FibGen(mesh, bdata, boundaries, apex_id, params)
lap, grad = fibgen.run_laplace_problems()
f, s, n = fibgen.get_fibers()
f_lin, s_lin, n_lin = fibgen.get_linear_fibers(method='bislerp')   # Just for comparison purposes

# Save results
save = np.hstack([f,s,n])
chio.write_dfile(data_path + 'fiber.field', save)
save = np.hstack([f_lin,s_lin,n_lin])
chio.write_dfile(data_path + 'fiber_lin.field', save)

# Save rv-lv division
rvlv = fibgen.run_lv_rv(gradient=False)
mesh.point_data['rvlv'] = rvlv
chio.write_dfile(data_path + 'rvlv.FE', rvlv)

# Save fiber topology
fib_mesh = fibgen.fib_mesh
chio.write_mesh(mesh_folder + 'fiber', fib_mesh.points, fib_mesh.cells[0].data)

# Visualization output
import meshio as io
data = fibgen.get_fiber_data()
fibgen.fib_mesh.point_data = data
io.write(data_path + 'fiber.vtu', fibgen.fib_mesh)


data = fibgen.get_point_data()
fibgen.mesh.point_data = data
fibgen.mesh.point_data['f'] = f_lin
fibgen.mesh.point_data['s'] = s_lin
fibgen.mesh.point_data['n'] = n_lin
io.write(data_path + 'fiber_data.vtu', fibgen.mesh)
