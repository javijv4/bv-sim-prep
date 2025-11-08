#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:01:37 2025

@author: Javiera Jilberto Vallejos
"""

import meshio as io
import cheartio as chio
import numpy as np

vol_mesh = io.read('bv_model.vtu')
chio.write_mesh('bv_model_gen', vol_mesh.points, vol_mesh.cells[0].data)

surf_mesh = io.read('bv_model_b.vtu')
boundaries = surf_mesh.cell_data['patches'][0]

labels = np.unique(boundaries)
cell_blocks = []
for l in labels:
    cell_block = surf_mesh.cells[0].data[boundaries == l]
    cell_blocks.append(io.CellBlock(cell_type='triangle', data=cell_block))

new_surf_mesh = io.Mesh(points=surf_mesh.points, 
                        cells=cell_blocks)

print(vol_mesh.points.shape, new_surf_mesh.points.shape)
bdata, _ = chio.create_bfile(new_surf_mesh, vol_mesh.cells[0].data, inner_faces=True)
chio.write_bfile('bv_model_gen', bdata)