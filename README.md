# DeepSphere: a spherical convolutional neural network

[Janis Fluri](http://www.da.inf.ethz.ch/people/JanisFluri),
[Nathanaël Perraudin](https://perraudin.info),
[Michaël Defferrard](https://deff.ch)

This is an implementation of DeepSphere using TensorFlow 2.x.

## Resources

Code:
* [deepsphere-cosmo-tf1](https://github.com/deepsphere/deepsphere-cosmo-tf1): original repository, implemented in TensorFlow v1. Use to reproduce [this paper][paper_cosmo].
* [deepsphere-cosmo-tf2](https://github.com/deepsphere/deepsphere-cosmo-tf2): reimplementation in TFv2. Use for new developments in TensorFlow targeting HEALPix.
* [deepsphere-tf1](https://github.com/deepsphere/deepsphere-tf1): extended to other samplings and experiments, implemented in TFv1. Use to reproduce [this paper][paper_iclr].
* [deepsphere-pytorch](https://github.com/deepsphere/deepsphere-pytorch): reimplementation in PyTorch. Use for new developments in PyTorch.

Papers:
* DeepSphere: Efficient spherical CNN with HEALPix sampling for cosmological applications, 2018.\
  [[paper][paper_cosmo], [blog](https://datascience.ch/deepsphere-a-neural-network-architecture-for-spherical-data), [slides](https://doi.org/10.5281/zenodo.3243380)]
* DeepSphere: towards an equivariant graph-based spherical CNN, 2019.\
  [[paper][paper_rlgm], [poster](https://doi.org/10.5281/zenodo.2839355)]
* DeepSphere: a graph-based spherical CNN, 2020.\
  [[paper][paper_iclr], [slides](https://doi.org/10.5281/zenodo.3777976), [video](https://youtu.be/NC_XLbbCevk)]

[paper_cosmo]: https://arxiv.org/abs/1810.12186
[paper_rlgm]: https://arxiv.org/abs/1904.05146
[paper_iclr]: https://openreview.net/forum?id=B1e3OlStPB

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/deepsphere/deepsphere-cosmo-tf2.git
   cd deepsphere-cosmo-tf2
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```
   **Note**: the code has been developed and tested with Python 3.6.
   It **does not** work on Python 2.7!

3. Install the package.
   ```sh
   pip install -e .
   ```

4. (Optional) Test the installation.
   ```
   pytest tests
   ```

5. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```

## Notebooks

The below notebooks contain examples and experiments to play with the model.

1. [Quick Start.][whole_sphere]
   The easiest to play with the model by classifying data on the whole sphere.
2. [Advanced Tutorial.][advanced]
   An introduction to various layers, customized training loops, and custom survey masks.
3. [Generative Models.][generative]
   How to build an auto-encoder using spherical data and the transpose healpy pseudo convolutions.

[whole_sphere]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-cosmo-tf2/blob/master/examples/quick_start.ipynb
[advanced]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-cosmo-tf2/blob/master/examples/advanced_tutorial.ipynb
[generative]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-cosmo-tf2/blob/master/examples/generative_models.ipynb

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).\
Please consider citing our papers if you find it useful.

```
@article{deepsphere_cosmo,
  title = {{DeepSphere}: Efficient spherical Convolutional Neural Network with {HEALPix} sampling for cosmological applications},
  author = {Perraudin, Nathana\"el and Defferrard, Micha\"el and Kacprzak, Tomasz and Sgier, Raphael},
  journal = {Astronomy and Computing},
  volume = {27},
  pages = {130-146},
  year = {2019},
  month = apr,
  publisher = {Elsevier BV},
  issn = {2213-1337},
  doi = {10.1016/j.ascom.2019.03.004},
  archiveprefix = {arXiv},
  eprint = {1810.12186},
  url = {https://arxiv.org/abs/1810.12186},
}
```

```
@inproceedings{deepsphere_rlgm,
  title = {{DeepSphere}: towards an equivariant graph-based spherical {CNN}},
  author = {Defferrard, Micha\"el and Perraudin, Nathana\"el and Kacprzak, Tomasz and Sgier, Raphael},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year = {2019},
  archiveprefix = {arXiv},
  eprint = {1904.05146},
  url = {https://arxiv.org/abs/1904.05146},
}
```

```
@inproceedings{deepsphere_iclr,
  title = {{DeepSphere}: a graph-based spherical {CNN}},
  author = {Defferrard, Michaël and Milani, Martino and Gusset, Frédérick and Perraudin, Nathanaël},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020},
  url = {https://openreview.net/forum?id=B1e3OlStPB},
}
```
