# asymmetry-and-machine-learning
Calculation of shape asymmetry statistic to identify ram pressure stripping galaxies. Machine learning is used to define morphological classes for a galaxy cluster.

Functions within asymetry_utils.py script are:

1. **create_segmentation()** --> To create a segmentation map of sources identified above some S/N threshold. The largest segementation is assumed to be our galaxy.

2. **calc_shape_asymetry()** --> Utilizing our segementation map, considers a fixed centre and radius to compute the shape asymmetry as described in [Pawlik et. al 2016](https://arxiv.org/abs/1512.02000). This is similar to [statmorph.shape_asymmetry](https://statmorph.readthedocs.io/en/latest/_modules/statmorph/statmorph.html#SourceMorphology) calculation, only we use a fixed centre. 
                              
3. **galaxy_cutout()** --> Same function found in radial-profile-utils repo. Creates thumbnail of our galaxy.

4. **MaskSources()** --> Same function found in radial-profile-utils repo. Masks sources in field of view along with edges and bad pixels. 
