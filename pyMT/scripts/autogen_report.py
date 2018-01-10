import pyMT.data_structures as WSDS
import pyMT.IO as WSIO
import pyMT.gplot as gplot
import glob
from numpy import log10
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os


def scan_models():
    models = []
    l_rms = 10000
    for file in glob.glob('*model*'):
        with open(file, 'r') as f:
            line = f.readline()
            if 'RMS' in line.upper():
                RMS = float(line.split('=')[1])
                if RMS > 0:
                    models.append((file, float(line.split('=')[1])))
                    if models[-1][1] < l_rms and models[-1][1] > 0:
                        best_model = (file, models[-1][1])
                        l_rms = best_model[1]
    models = sorted(models, key=lambda x: x[1])
    return best_model, models


def write_template(dataset, startup, outfile):
    with open(outfile, 'w') as f:
        f.write('Project Name: \n')
        f.write('Directory: \n')
        f.write('Site list: {} \n'.format(dataset.raw_data.listfile))
        f.write('Data file: {} \n'.format(dataset.data.datafile))
        f.write('Model file: {} \n'.format(dataset.model.file))
        f.write('Response file: {} \n'.format(dataset.response.datafile))
        f.write('Inversion type: {} \n'.format(startup['inv_type']))
        f.write('Azimuth: {}\n'.format(dataset.data.azimuth))
        f.write('Number of sites: {}\n'.format(dataset.data.NS))
        f.write('Impedance error floor: {}%\n'.format(startup['errFloorZ']))
        if startup['inv_type'] == 5:
            f.write('Tipper error floor: {}%\n'.format(startup['errFloorT']))
        RMS = 0.
        f.write('Starting half-space rho: \n')
        worst_site = ''
        for (site, info) in dataset.rms['Station'].items():
            if info['Total'] > RMS:
                RMS = info['Total']
                worst_site = site
        # worst_site = min(dataset.rms['Station'], key=dataset.rms['Station']['Total'].get)
        f.write('Final RMS: {:.5f} \n'.format(dataset.rms['Total']))
        f.write('Worst site misfit: {} , RMS = {:.5f} \n'.format(worst_site, RMS))
        f.write('{} Inverted Periods: \n'.format(dataset.data.NP))
        f.write('{:>20}{:>20}{:>20}\n'.format('Period (s)', 'Frequency (Hz)', 'Log10(Period)'))
        for p in dataset.data.periods:
            f.write('{:20.5f}{:20.5f}{:20.5f}\n'.format(p, 1 / p, log10(p)))
        f.write('Notes:')


def confirm_details(best_model, models):
    print('Best model is {} with an RMS of {}'.format(best_model[0], best_model[1]))
    resp = WSIO.verify_input('This is the correct model?', expected='yn', default='y')
    if resp == 'n':
        print('Options are:')
        for model in models:
            print('\t{}  RMS: {}'.format(model[0], model[1]))
        model = WSIO.verify_input(message='New selection', expected='read', default=best_model[0])
    else:
        model = best_model[0]
    listfile = WSIO.verify_input(message='Enter list file:', expected='read', default=None)
    outfile = WSIO.verify_input(message='Output file name?', expected='write', default='Report.txt')
    return model, listfile, outfile


def generate_misfit_plots(dataset):
    if not os.path.exists('Report'):
        os.makedirs('Report')
    if not os.path.exists('Report/misfit_curves'):
            os.makedirs('Report/misfit_curves')
    path = 'Report/misfit_curves/'
    fig = plt.figure(figsize=(26, 15))
    dpm = gplot.DataPlotManager(fig=fig)
    components = {'impedance_diagonal': ('ZXXR', 'ZXXI', 'ZYYR', 'ZYYI'),
                  'impedance_off-diagonal': ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI'),
                  'phase_diagonal': ('PhaXX', 'PhaYY'),
                  'rho_diagonal': ('RhoXX', 'RhoYY'),
                  'rho_off-diagonal': ('RhoXY', 'RhoYX'),
                  'phase_off-diagonal': ('PhaXY', 'PhaYX'),
                  'tipper': ('TZXR', 'TZXI', 'TZYR', 'TZYI')}
    dpm.show_outliers = False
    dpm.markersize = 6
    dpm.outlier_thresh = 5
    if 'TZXR' not in dataset.data.components:
        del components['tipper']
    for comps in components.keys():
        if 'phase' in comps or 'rho' in comps:
            dpm.link_axes_bounds = True
        else:
            dpm.link_axes_bounds = False
        print('Generating {} plots...'.format(comps))
        dpm.components = components[comps]
        for ii in range(0, len(dataset.data.site_names), 6):
            if comps == 'tipper':
                dpm.scale = 'none'
            else:
                dpm.scale = 'sqrt(periods)'
            sites = dataset.data.site_names[ii:ii + 6]
            dpm.sites = dataset.get_sites(site_names=sites, dTypes='all')
            dpm.plot_data()
            if ii == 0:
                pp = PdfPages(''.join([path, comps, '_misfit.pdf']))
            dpm.fig.savefig(pp, format='pdf', dpi=1000, bbox_inches='tight')
            # dpm.fig.savefig(''.join([comps, '_misfit', str(ii), '.pdf']), bbox_inches='tight', dpi=1000)
        pp.close()


def main():
    best_model, models = scan_models()
    startup = WSIO.read_startup()
    model, listfile, outfile = confirm_details(best_model, models)
    response = model.replace('model', 'resp')
    dataset = WSDS.Dataset(modelfile=model, datafile=startup['datafile'],
                           responsefile=response, listfile=listfile)
    write_template(dataset, startup, outfile)
    generate_misfit_plots(dataset)


if __name__ == '__main__':
    main()
