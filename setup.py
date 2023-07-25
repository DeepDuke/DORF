import setuptools


setuptools.setup(
    name='dorf',
    version='0.0.1',
    description='DORF: Dynamic Object Removal Framework',
    packages=['dorf', 'dorf.filters', 'dorf.utils'],
    author='deepduke',
    setup_requires=[
        'wheel',
        'numpy',
        'scikit-learn',
        'pypcd>=0.1.1',
        'bresenham>=0.2'
        ]
    )
        
