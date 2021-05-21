from setuptools import setup

setup(
    name='deep_surfel',
    version='1.0.0',
    packages=['deep_surfel', 'deep_surfel.lib'],
    url='https://onlinereconstruction.github.io/DeepSurfels',
    author='Marko Mihajlovic',
    author_email='markomih@ethz.ch',
    description='DeepSurfels is a novel 3D representation for geometry and appearance information '
                'that combines planar surface primitives with voxel grid representation for improved '
                'scalability and rendering quality.'
)
