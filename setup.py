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
                'pyMT.scripts'],
      long_description=open('README.txt').read(),
      scripts=['pymt/scripts/autogen_report.py',
               'pymt/scripts/plot_dataset.py',
               'pymt/scripts/plot_occam.py',
               'pymt/scripts/j2ws3d.py',
               'pymt/scripts/plot_transect.py',
               'pymt/scripts/to_vtk.py',
               'pymt/scripts/ws2modem.py',
               'pymt/DataGUI/data_plot.py',
               'pymt/ModelGUI/mesh_designer.py'])
