#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 07:42:57 2023

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
import meshio as io
import cheartio as chio

simmodeler_path = 'test_data/'
mesh_path = 'test_data/mesh/'
model_name = 'bv_model'          # output name for CH files
vol_mesh_name = simmodeler_path + 'volume'
surf_mesh_name = simmodeler_path + 'surface'
swap = True

# Create output directory
if not os.path.exists(mesh_path): os.mkdir(mesh_path)

# Reading .inp
vol_mesh = io.read(vol_mesh_name + '.inp')
surf_mesh = io.read(surf_mesh_name + '.inp')

# Join RV and LV regions
# The LV should be the region with more elements
sizes = [len(vol_mesh.cells[0]), len(vol_mesh.cells[1])]
lv_region = np.argmax(sizes)            # Assumes the region with more elements is the LV
rv_region = 1-lv_region
cells = np.vstack([vol_mesh.cells[lv_region].data, vol_mesh.cells[rv_region].data])
region = np.zeros(len(cells), dtype=int)
region[0:sizes[lv_region]] = 1

# Generate boundary file
bdata, _ = chio.create_bfile(surf_mesh, cells, inner_faces = True)

# Write CH files
chio.write_mesh(mesh_path + model_name, vol_mesh.points, cells)
chio.write_dfile(mesh_path + 'region.FE', region)

# Write .vtu
mesh = io.Mesh(vol_mesh.points, {'tetra': cells}, cell_data = {'region': [region]})
io.write(mesh_path + model_name + '.vtu', mesh)

bmesh = io.Mesh(vol_mesh.points, {'triangle': bdata[:,1:-1]},
                cell_data = {'patches': [bdata[:,-1]]})
io.write(mesh_path + model_name + '_boundary.vtu', bmesh)

# Identify patches
xyz = mesh.points
ien = mesh.cells[0].data
labels, cont = np.unique(bdata[:,-1], return_counts=True)
labels = labels[np.argsort(cont)[::-1]]  # Sort by number of elements

# Last four are valves
valve_labels = labels[-4:]
valve_nodes = [np.unique(bdata[bdata[:,-1] == label][:,1:-1]) for label in valve_labels]
valve_centroids = np.array([np.mean(xyz[nodes], axis=0) for nodes in valve_nodes])

# MV has the most elements
mv_label = valve_labels[0]  
mv_nodes = np.unique(bdata[bdata[:,-1] == mv_label][:,1:-1])
mv_centroid = valve_centroids[0]
valve_labels = np.delete(valve_labels, 0)  # Remove MV from the list
valve_centroids = np.delete(valve_centroids, 0, axis=0)  # Remove MV from the list

# AV is the closest to the MV
distances = [np.linalg.norm(mv_centroid - centroid) for centroid in valve_centroids[1:]]
av_label = valve_labels[np.argmin(distances) + 1]
av_nodes = np.unique(bdata[bdata[:,-1] == av_label][:,1:-1])
av_centroid = valve_centroids[np.argmin(distances) + 1]
valve_labels = np.delete(valve_labels, np.argmin(distances) + 1)  # Remove AV from the list
valve_centroids = np.delete(valve_centroids, np.argmin(distances) + 1, axis=0)  # Remove AV from the list

# Apex is the LV node farthest from the MV
lv_nodes = np.unique(ien[region == 1])
distances = np.linalg.norm(xyz[lv_nodes] - mv_centroid, axis=1)
apex_node = lv_nodes[np.argmax(distances)]

# LA vector
la_vector = mv_centroid - xyz[apex_node]
la_vector /= np.linalg.norm(la_vector)

# Septum vector
septum_vector = av_centroid - mv_centroid
septum_vector /= np.linalg.norm(septum_vector)

# Third vector
third_vector = np.cross(la_vector, septum_vector)
third_vector /= np.linalg.norm(third_vector)

# Calculate projected distance to the third vector for the valves
projected_distances = np.dot(valve_centroids - mv_centroid, third_vector)

# PV should have the most positive distance
pv_label = valve_labels[np.argmax(projected_distances)]
pv_nodes = np.unique(bdata[bdata[:,-1] == pv_label][:,1:-1])
pv_centroid = valve_centroids[np.argmax(projected_distances)]

# TV should have the most negative distance
tv_label = valve_labels[np.argmin(projected_distances)]
tv_nodes = np.unique(bdata[bdata[:,-1] == tv_label][:,1:-1])
tv_centroid = valve_centroids[np.argmin(projected_distances)]

valve_labels = np.array([mv_label, av_label, pv_label, tv_label])

# Finding the rest
labels, cont = np.unique(bdata[:,-1], return_counts=True)
patch_nodes = [np.unique(bdata[bdata[:,-1] == label][:,1:-1]) for label in labels]

neigh_matrix = np.zeros((len(patch_nodes), len(patch_nodes)), dtype=int)
for i in range(len(patch_nodes)):
    for j in range(i+1, len(patch_nodes)):
        neigh_matrix[i,j] = len(np.intersect1d(patch_nodes[i], patch_nodes[j]))
        neigh_matrix[j,i] = neigh_matrix[i,j]

neigh_matrix = neigh_matrix.astype(bool)

# Looking for the LV endo (it has only two neighbors and one of them is the MV)
for i, l in enumerate(labels):
    if l in valve_labels:
        continue
    neigh = neigh_matrix[i]
    num_neigh = np.sum(neigh)
    if num_neigh > 2:
        continue
    if neigh[labels==mv_label] and neigh[labels==av_label]: # If neighbor with the MV
        # This is the LV endo
        lvendo_label = labels[i]
        lvendo_nodes = patch_nodes[i]
        lvendo_centroid = np.mean(xyz[lvendo_nodes], axis=0)
        break 
    

# LV epi is the patch that has the closest centroid to the LV centroid
patches_centroids = np.array([np.mean(xyz[nodes], axis=0) for nodes in patch_nodes])   # Remove valves
distances = np.linalg.norm(patches_centroids - lvendo_centroid, axis=1)
lvepi_label = labels[np.argsort(distances)[1]]  # +1 to skip the LV endo
lvepi_nodes = patch_nodes[np.argmin(distances)]
lvepi_centroid = np.mean(xyz[lvepi_nodes], axis=0)
identified_patches = np.concatenate([valve_labels, [lvendo_label, lvepi_label]])

# RV epi is neighbor of the LV epi and the PV and TV
available_patches = np.delete(labels, np.where(np.isin(labels, identified_patches)))
neigh = neigh_matrix[available_patches-1][:, np.array([lvepi_label-1, pv_label-1, tv_label-1])]

rvepi_label = available_patches[np.where(np.sum(neigh, axis=1) == 3)[0][0]]
rvepi_nodes = patch_nodes[np.where(labels == rvepi_label)[0][0]]
rvepi_centroid = np.mean(xyz[rvepi_nodes], axis=0)
identified_patches = np.concatenate([identified_patches, [rvepi_label]])

# RV sep is not neighbot with any of the above
rvsep_label = available_patches[np.where(np.sum(neigh, axis=1) == 0)[0][0]]
rvsep_nodes = patch_nodes[np.where(labels == rvsep_label)[0][0]]
rvsep_centroid = np.mean(xyz[rvsep_nodes], axis=0)
identified_patches = np.concatenate([identified_patches, [rvsep_label]])

# RV endo is the patch that is neighbor with the valves
rvendo_label = available_patches[np.where(np.sum(neigh, axis=1) == 2)[0][0]]
rvendo_nodes = patch_nodes[np.where(labels == rvendo_label)[0][0]]
rvendo_centroid = np.mean(xyz[rvendo_nodes], axis=0)
identified_patches = np.concatenate([identified_patches, [rvendo_label]])

# RV-LV junction is only neighbor with LV epi
rvlv_label = available_patches[np.where(np.sum(neigh, axis=1) == 1)[0][0]]
rvlv_nodes = patch_nodes[np.where(labels == rvlv_label)[0][0]]
rvlv_centroid = np.mean(xyz[rvlv_nodes], axis=0)
identified_patches = np.concatenate([identified_patches, [rvlv_label]])

# Swap labels
if swap:
    swap_labels = {lvendo_label: 1,   # endo lv
                   rvendo_label: 2,   # endo rv
                   lvepi_label: 3,   # epi lv
                   rvepi_label: 4,   # epi rv
                   rvsep_label: 5,   # rv septum
                   av_label: 6,   # av
                   pv_label: 7,   # pv
                   tv_label: 8,   # tv
                   mv_label: 9,   # mv
                   rvlv_label: 10}   # junction

    new_labels = np.zeros(len(bdata))
    labels = bdata[:,-1]

    for i, s in enumerate(swap_labels.keys()):
        new_labels[labels == s] = i+1

    bdata[:,-1] = new_labels

    # Write CH files
    chio.write_bfile(mesh_path + model_name, bdata)

    bmesh = io.Mesh(vol_mesh.points, {'triangle': bdata[:,1:-1]},
                    cell_data = {'patches': [bdata[:,-1]]})
    io.write(mesh_path + model_name + '_boundary.vtu', bmesh)

    labels = {
                'lv_endo': 1,
                'rv_endo': 2,
                'lv_epi': 3,
                'rv_epi': 4,
                'rv_septum': 5,
                'av': 6,
                'pv': 7,
                'tv': 8,
                'mv': 9,
                'rvlv_junction': 10,
                'apex': apex_node,
            }

else:
    # Save dictionary with info
    labels = {
                'lv_endo': lvendo_label,
                'rv_endo': rvendo_label,
                'lv_epi': lvepi_label,
                'rv_epi': rvepi_label,
                'rv_septum': rvsep_label,
                'av': av_label,
                'pv': pv_label,
                'tv': tv_label,
                'mv': mv_label,
                'rvlv_junction': rvlv_label,
                'apex': apex_node,
            }

chio.dict_to_pfile(mesh_path + 'boundaries.P', labels)