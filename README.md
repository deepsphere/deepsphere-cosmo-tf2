# DeepSphere TF 2.x: a spherical convolutional neural network

[Janis Fluri][jafluri], [Nathanaël Perraudin][nath], [Michaël Defferrard][mdeff]

[jafluri]: http://www.da.inf.ethz.ch/people/JanisFluri
[nath]: https://perraudin.info
[mdeff]: http://deff.ch

This is an implementation of the original [Deepsphere repository](https://github.com/deepsphere/DeepSphere) using Tensorflow 2.x.

Ressources:
* **blog post**: [DeepSphere: a neural network architecture for spherical data][blog]
* **paper (cosmo)**: [DeepSphere: Efficient spherical CNN with HEALPix sampling for cosmological applications][paper_cosmo]
* **paper (ML)**: [DeepSphere: a graph-based spherical CNN with approximate equivariance][paper_ml]
* **slides**: [DeepSphere: Efficient spherical CNN with HEALPix sampling for cosmological applications][slides] ([AIcosmo2019])

[blog]: https://datascience.ch/deepsphere-a-neural-network-architecture-for-spherical-data
[paper_cosmo]: https://arxiv.org/abs/1810.12186
[paper_ml]: https://arxiv.org/abs/1904.05146
[slides]: https://doi.org/10.5281/zenodo.3243380
[AIcosmo2019]: https://sites.google.com/site/aicosmo2019

[gcnn_paper]: https://arxiv.org/abs/1606.09375
[gcnn_code]: https://github.com/mdeff/cnn_graph/

## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/deepsphere/DeepSphere_TF2
   cd DeepSphere_TF2
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```
   **Note**: the code has been developed and tested with Python 3.5 and 3.6.
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

[whole_sphere]: examples/quick_start.ipynb
[advanced]: examples/advanced_tutorial.ipynb
[generative]: examples/generative_models.ipynb

## License & citation

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).
Please consider citing our papers if you use it.

```
@article{deepsphere_cosmo,
  title = {DeepSphere: Efficient spherical Convolutional Neural Network with HEALPix sampling for cosmological applications},
  author = {Perraudin, Nathana\"el and Defferrard, Micha\"el and Kacprzak, Tomasz and Sgier, Raphael},
  journal = {Astronomy and Computing},
  year = {2018},
  archivePrefix = {arXiv},
  eprint = {1810.12186},
  url = {https://arxiv.org/abs/1810.12186},
}
```

```
@inproceedings{deepsphere_ml,
  title = {DeepSphere: towards an equivariant graph-based spherical CNN},
  author = {Defferrard, Micha\"el and Perraudin, Nathana\"el and Kacprzak, Tomasz and Sgier, Raphael},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year = {2019},
  archivePrefix = {arXiv},
  eprint = {1904.05146},
  url = {https://arxiv.org/abs/1904.05146},
}
```
