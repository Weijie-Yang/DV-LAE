from setuptools import setup, find_packages

setup(
    name='DV_LAE',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'ase==3.22.1',
        'numpy==1.26.3',
        'plotly==5.18.0',
        'scikit-learn==1.4.0',
        'tqdm==4.66.1'
    ],
)
