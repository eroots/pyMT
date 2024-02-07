from setuptools import setup


setup(name='pyMT',
      version='0.102',
      description='Tools and GUIs for MT data analysis and modelling',
      url='http://github.com/eroots/pyMT',
      author='Eric Roots',
      author_email='eroots087@gmail.com',
      packages=['pyMT',
                'pyMT.DataGUI',
                'pyMT.ModelGUI',
                'pyMT.GatewayGUI',
                'pyMT.scripts',
                'pyMT.GUI_common',
                'pyMT.e_colours',
                'pyMT.resources'],
      long_description=open('readme.rst').read(),
      scripts=['pyMT/scripts/autogen_report.py',
               'pyMT/scripts/plot_dataset.py',
               'pyMT/scripts/plot_occam.py',
               'pyMT/scripts/j2ws3d.py',
               'pyMT/scripts/plot_transect.py',
               'pyMT/scripts/to_vtk.py',
               'pyMT/scripts/ws2modem.py',
               'pyMT/scripts/f2l.py'],
      entry_points={'console_scripts': ['data_plot = pyMT.DataGUI.data_plot:main',
                                        'mesh_designer = pyMT.ModelGUI.mesh_designer:main',
                                        'model_viewer = pyMT.ModelGUI.model_viewer:main',
                                        'gateway_mt = pyMT.GatewayGUI.gateway_main:main']},
      install_requires=['numpy',
                        'scipy',
                        'colorcet',
                        'matplotlib',
                        'https://github.com/eroots/natural-neighbor-interpolation',
                        'pyproj',
                        'pyshp',
                        'pyvista',
                        'pyvistaqt'
                        'pyqt5',
                        'vtk'],
      include_package_data=True)
