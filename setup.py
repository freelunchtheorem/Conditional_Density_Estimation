from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='cde',
      version='1.0',
      long_description=long_description,
      long_description_content_type='text/markdown',
      description='Framework for conditional density estimation',
      url='https://github.com/freelunchtheorem/Conditional_Density_Estimation',
      author='Jonas Rothfuss, Fabio Ferreira',
      author_email='jonas.rothfuss@gmx.de, fabioferreira@mailbox.org',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'seaborn',
        'pytest',
        'scikit_learn',
        'statsmodels',
        'wandb',
      ],
      zip_safe=False)
