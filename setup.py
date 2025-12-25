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
        'ml_logger<=99.99',
        'progressbar2',
        'xlrd'
      ],
      dependency_links=["https://github.com/jonasrothfuss/ml_logger/archive/2000b38177e3c4892e4fee74d769c1fc0a659424.zip#egg=ml_logger-99.99"],
      zip_safe=False)
