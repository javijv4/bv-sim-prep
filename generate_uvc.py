#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:17:37 2023

@author: Javiera Jilberto Vallejos
"""
import sys
import cheartio as chio
from uvcgen.model_coords import UVCGen
from uvcgen.UVC import UVC
from uvcgen.uvc_outputs import export_info, export_cheart_inputs

# Inputs
mesh_folder = 'test_data/mesh/'
region_split_file = mesh_folder + 'region.FE'
mesh_path = mesh_folder + 'bv_model'


# load boundaries and apex
boundaries = chio.pfile_to_dict(mesh_folder + 'boundaries.P', fmt=int)
apex_id = boundaries['apex']  # lv apex point in the epicardium

thresholds = {'septum'        : 0.6,        # Splits RV from LV.
              'long'          : 1.0}
method = 'laplace'

# Read CH mesh
print('Initializing')
bv_mesh = chio.read_mesh(mesh_path, meshio=True)
bdata = chio.read_bfile(mesh_path)
try:
    rvlv = 1 - chio.read_dfile(region_split_file)   # This file is 0 in LV, 1 in RV
except:
    rvlv = None

# Initialize classes
uvc = UVC(bv_mesh, bdata, boundaries, thresholds, mesh_folder, rvlv=rvlv)
mcg = UVCGen(uvc, mmg=True)
if not uvc.split_epi:
    septum = mcg.run_septum(uvc)
else:
    septum = uvc.compute_septum()
uvc.compute_long_plane_coord(septum)
uvc.split_rv_lv(septum)

# Compute coordinates
uvc.define_apex_nodes()
print('Computing Transmural')
lv_trans, rv_trans = mcg.run_transmural(uvc, method='laplace')
print('Computing Longitudinal')
long = mcg.run_longitudinal(uvc)
uvc.merge_lv_rv_point_data(['long'])
uvc.define_septum_nodes()
print('Computing Circumferential')
lv_circ, rv_circ = mcg.run_circumferential(uvc)
if mcg.mmg:
    print('Correcting Longitudinal')
    long = mcg.correct_longitudinal(uvc)

print('Postprocessing ')
uvc.rv_mesh.point_data['long'] = uvc.bv_mesh.point_data['long'][uvc.map_rv_bv]
mcg.get_local_vectors(uvc, which='lv')
mcg.get_local_vectors(uvc, which='rv')

print('Computing AHA segments')
uvc.compute_aha_segments(aha_type='points')
uvc.compute_aha_segments(aha_type='elems')

# Collecting results into BV mesh
uvc.merge_lv_rv_point_data(['circ', 'trans', 'circ_aux', 'long', 'aha', 'eC', 'eL', 'eT'])
uvc.merge_lv_rv_cell_data(['aha'])

import meshio as io
io.write(mesh_folder + 'lv_mesh.vtu', uvc.lv_mesh)
io.write(mesh_folder + 'rv_mesh.vtu', uvc.rv_mesh)
io.write(mesh_folder + 'bv_mesh.vtu', uvc.bv_mesh)

export_info(uvc)
export_cheart_inputs(uvc)
