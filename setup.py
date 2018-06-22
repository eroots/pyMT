from setuptools import setup


setup(name='pyMT',
      version='0.1',
      description='Tools and GUIs for MT data analysis and modelling',
      url='http://github.com/eroots/pyMT',
      author='Eric Roots',
      author_email='eroots087@gmail.com',
      packages=['pyMT',
                'pyMT.DataGUI',
                'pyMT.ModelGUI',
                'pyMT.scripts',
                'pyMT.GUI_common'],
      long_description=open('readme.txt').read(),
      scripts=['pyMT/scripts/autogen_report.py',
               'pyMT/scripts/plot_dataset.py',
               'pyMT/scripts/plot_occam.py',
               'pyMT/scripts/j2ws3d.py',
               'pyMT/scripts/plot_transect.py',
               'pyMT/scripts/to_vtk.py',
               'pyMT/scripts/ws2modem.py',
               'pyMT/scripts/f2l.py',
               'pyMT/scripts/convert_edi.py'],
      entry_points={'gui_scripts': 'data_plot = pyMT.DataGUI.data_plot:main'})
