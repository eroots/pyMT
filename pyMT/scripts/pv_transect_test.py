import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
import pyMT.data_structures as WSDS
import pyvista as pv

n_interp = 200


def interpolate_between_points(E, N):
    qx = []
    qy = []
    distance = []
    for ii in range(len(E) - 1):
        distance.append(np.sqrt((E[ii] - E[ii + 1]) ** 2 +
                                (N[ii] - N[ii + 1]) ** 2))
    distance = np.array(distance)
    total_distance = np.sum(distance)
    N_per_segment = np.ceil(n_interp * distance / total_distance)
    for ii in range(len(E) - 1):
        # E1, E2 = (min(E[ii], E[ii + 1]), max(E[ii], E[ii + 1]))
        if E[ii] > E[ii + 1]:
            E0, E1 = E[ii + 1], E[ii]
            N0, N1 = N[ii + 1], N[ii]
        else:
            E0, E1 = E[ii], E[ii + 1]
            N0, N1 = N[ii], N[ii + 1]
        X = np.linspace(E0, E1, N_per_segment[ii])
        Y = np.interp(X, (E0, E1), (N0, N1))
        if E[ii] > E[ii + 1]:
            X = np.flip(X, axis=0)
            Y = np.flip(Y, axis=0)
        qx.append(X)
        qy.append(Y)
    qx = np.concatenate(qx).ravel()
    qy = np.concatenate(qy).ravel()
    # debug_print((qx, qy), 'debugT.log')
    return qx, qy


def generate_transect2D(transect_plot, clip_model):
    if not (transect_plot['easting'] and transect_plot['northing']):
        return
    qx, qy = interpolate_between_points(transect_plot['easting'],
                                        transect_plot['northing'])
    # qz = np.array(clip_model.dz[20:50])
    qz = np.array(clip_model.dz)
    query_points = np.zeros((len(qx) * len(qz), 3))
    faces = np.zeros(((len(qx) - 1) * (len(qz) - 1), 5))
    cc = 0
    cc2 = 0
    for ix in range(len(qx)):
        for ii, iz in enumerate(qz):
            query_points[cc, :] = np.array((qx[ix], qy[ix], iz))
            if ii < len(qz) - 1 and ix < len(qx) - 1:
                faces[cc2, 0] = 4
                faces[cc2, 1] = cc
                faces[cc2, 2] = cc + 1
                faces[cc2, 3] = len(qz) + cc + 1
                faces[cc2, 4] = len(qz) + cc
                cc2 += 1
            cc += 1
    # cc = 0
    # for iz in qz:
    #     for ix in range(len(qx)):
    #         query_points[cc, :] = np.array((qx[ix], qy[ix], iz))
    #         cc += 1
    cell_N, cell_E, cell_z = clip_model.cell_centers()
    vals = np.transpose(np.log10(clip_model.vals), [1, 0, 2])
    interpolator = RGI((cell_E, cell_N, cell_z),
                       vals, bounds_error=False, fill_value=5)
    interp_vals = interpolator(query_points)
    # map.window['axes'][0].plot(qx, qy, 'k--')
    # canvas['2D'].draw()
    # debug_print
    interp_vals = np.reshape(interp_vals, [len(qx), len(qz)])
    return interp_vals, query_points, faces


def generate_transect3D(query_points, faces):
        query_points[:, 2] *= -1
        poly = pv.PolyData(query_points, faces)
        # self.slices['transect'] = poly.delaunay_2d()
        return poly


# def plot_transect3D(redraw=False):    
#     self.vtk_widget.remove_actor(self.actors['transect'])
#     # self.slices['transect'] = self.generate_slice('z')
#     self.actors['transect'] = self.vtk_widget.add_mesh(self.slices['transect'])


model = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swzFinish_lastIter.rho')
model.spatial_units = 'km'
transect_plot = {'easting': [-50, 0, 50], 'northing': [-50, 0, 100]}
interp_vals, query_points, faces = generate_transect2D(transect_plot, model)
# poly = generate_transect3D(query_points)
# poly['labels'] = [x for x in range(len(query_points))]
# # query_points[:, [2, 1]] = query_points[:, [1, 2]]
poly = generate_transect3D(query_points, faces)
# # surf = poly.delaunay_2d(tol=0.1, alpha=10000, offset=0.100, bound=False)
plotter = pv.Plotter()
# plotter.add_point_labels(poly, 'labels')
# plotter.show_grid()
# plotter.show()
# plotter.add_mesh(poly, show_edges=True) #, scalars=interp_vals)
plotter.add_mesh(poly, show_edges=False, scalars=interp_vals)
plotter.show_grid()
plotter.show()

# vertices = np.array([[0, 0, 0],
#                      [1, 0, 0],
#                      [1, 1, 0],
#                      [0, 1, 0],
#              [0.5, 0.5, -1]])

# # mesh faces
# faces = np.hstack([[4, 0, 1, 2, 3],  # square
#                    [3, 0, 1, 4],     # triangle
#                    [3, 1, 2, 4]])    # triangle

# surf = pv.PolyData(vertices, faces)

# # plot each face with a different color
# surf['labels'] = [x for x in range(len(surf.points))]
# plotter.add_mesh(surf)
# plotter.add_point_labels(surf, 'labels')
# plotter.show()
# surf.plot(scalars=np.arange(3), cpos=[-1,1,0.5])