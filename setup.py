from setuptools import setup, find_packages

setup(
    name="deepsphere",
    version="0.1",
    description='DeepSphere in TF 2.x for healpy.',
    author='Janis Fluri',
    author_email='janis.fluri@phys.ethz.ch',
    python_requires='>=3.5, <4',
    keywords='healpy, graph convolutions',
    packages=find_packages(include=["deepsphere.*"]),
    project_urls={'DeepSphere': 'https://github.com/deepsphere'},
)
