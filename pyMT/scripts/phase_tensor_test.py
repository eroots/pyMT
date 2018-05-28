import numpy as np
import pyMT.data_structures as WSDS


def form_tensors(site):
    X, Y = [], []
    for ii in range(len(site.periods)):
        X.append(np.array(((site.data['ZXXR'][ii], site.data['ZXYR'][ii]),
                           (site.data['ZYXR'][ii], site.data['ZYYR'][ii]))))
        Y.append(np.array(((site.data['ZXXI'][ii], site.data['ZXYI'][ii]),
                           (site.data['ZYXI'][ii], site.data['ZYYI'][ii]))))
    return X, Y


def calculate_phase_parameters(phi):
    det_phi = np.linalg.det(phi)
    skew_phi = (phi[0, 1] - phi[1, 0])
    phi_1 = np.trace(phi) / 2
    phi_2 = np.lib.scimath.sqrt(det_phi)
    phi_3 = skew_phi / 2
    phi_max = np.sqrt(phi_1 ** 2 + phi_3 ** 2) + np.sqrt(phi_1 ** 2 + phi_3 ** 2 - det_phi)
    phi_min = np.sqrt(phi_1 ** 2 + phi_3 ** 2) - np.sqrt(phi_1 ** 2 + phi_3 ** 2 + det_phi)
    alpha = 0.5 * np.arctan((phi[0, 1] + phi[1, 0]) / (phi[0, 0] - phi[1, 1]))
    Lambda = (phi_max - phi_min) / (phi_max + phi_min)
    beta = 0.5 * np.arctan(phi_3 / phi_1)
    azimuth = 0.5 * np.pi - (alpha - beta)
    parameters = {'det_phi': det_phi,
                  'skew_phi': skew_phi,
                  'phi_1': phi_1,
                  'phi_2': phi_2,
                  'phi_3': phi_3,
                  'phi_max': phi_max,
                  'phi_min': phi_min,
                  'alpha': np.rad2deg(alpha),
                  'beta': np.rad2deg(beta),
                  'lambda': Lambda,
                  'azimuth': np.rad2deg(azimuth)}
    return parameters


def calculate_phase_tensor(site):
    X, Y = form_tensors(site)
    # phi = [np.matmul(np.linalg.inv(x), y) for x, y in zip(X, Y)]
    phi = np.matmul(np.linalg.inv(X), Y)
    return phi, X, Y


if __name__ == '__main__':
    filename = 'F:/GJH/TNG&MTR-EDI/all.lst'
    data = WSDS.RawData(filename)
    site = data.sites[data.site_names[0]]
    all_phi, X, Y = calculate_phase_tensor(site)
    phi_parameters = {}
    for ii, phi in enumerate(all_phi):
        parameters = calculate_phase_parameters(phi)
        if ii == 0:
            phi_parameters = {key: np.zeros((len(site.periods))) for key in parameters.keys()}
            phi_parameters['phi_2'] = np.zeros((len(site.periods)), 'complex')
        for key, value in parameters.items():
            phi_parameters[key][ii] = value
