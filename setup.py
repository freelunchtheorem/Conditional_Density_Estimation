from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='cde',
      version='0.5.1',
      long_description=long_description,
      description='Framework for conditional density estimation',
      url='https://jonasrothfuss.github.io/Nonparametric_Density_Estimation',
      author='Jonas Rothfuss, Fabio Ferreira',
      author_email='jonas.rothfuss@gmx.de, fabioferreira@mailbox.org',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'numpy>=1.26.4',
        'pandas',
        'matplotlib',
        'scipy>=1.11.3',
        'seaborn',
        'pytest',
        'scikit_learn',
        'statsmodels',
        'wandb',
        'progressbar2',
        'xlrd'
      ],
      zip_safe=False)
