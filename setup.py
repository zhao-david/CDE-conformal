from setuptools import setup

def readme():
    with open("README.rst") as f:
        return f.read()

setup(name='local_conformal',
      version='0.1',
      description='Local conformal inference with CDEs based on fit diagnostics',
      long_description = readme(),
      url='https://github.com/zhao-david/CDE-conformal',
      author='David Zhao, Benjamin LeRoy',
      author_email='dzhaoism@gmail.com, benjaminpeterleroy@gmail.com',
      license='MIT',
      packages=['local_conformal'],
      install_requires=[
          'numpy', 'pandas', 'scipy', # base libraries
          'progressbar2', 'matplotlib', # this line is less "needed"
          'cde_diagnostics', # david's package
          'torch', 'torch-distributions', # pytorch
          'plotnine', 'io', 'PIL', 'warnings' # for useful plotnine extensions
      ],
      test_require=['pytest'],
      zip_safe=False)
