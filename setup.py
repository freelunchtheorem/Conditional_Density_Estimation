from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='cde',
      version='0.2.0.1',
      long_description=long_description,
      description='Framework for conditional density estimation',
      url='https://jonasrothfuss.github.io/Nonparametric_Density_Estimation',
      author='Jonas Rothfuss, Fabio Ferreira',
      author_email='jonas.rothfuss@gmx.de, fabioferreira@mailbox.org',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'Keras==2.1.2',
        'numpy>=1.13.3',
        'pandas>=0.21.0',
        'tensorflow==1.4.1',
        'matplotlib>=2.1.0',
        'edward==1.3.4',
        'seaborn',
        'scipy>=1.0.0',
        'pytest>=3.3.2',
        'scikit_learn>=0.19.1',
        'dask>=1.0.0',
        'statsmodels',
        'pypmc',
        'ml_logger<=99.99',
      ],
      dependency_links=["https://github.com/jonasrothfuss/ml_logger/archive/2d373835ea159587fc323140ed0e8a8ea1bf9843.zip#egg=ml_logger-99.99"],
      zip_safe=False)
