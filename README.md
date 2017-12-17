This repository contains code for preprocessing the
[LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI),
a publicly available collection of annotated lung cancer screening CT scans.

## Stages

The preprocessing code is broken up into two stages:

* `stage1` extracts image data and annotations from the DICOM and XML files
  from the original dataset. The CT scans are represented as volumetric data
  after this stage.
* `stage2` extracts axial, sagittal, and coronal slices from the `stage1` output.
  This stage also produces augmented views of each example. The CT scans are
  represented as 2D images after this stage.

## Citations

If you use this preprocessing code in your research, we would greatly
appreciate you citing the following paper:

```bibtex
@article{nibali2017pulmonary,
  author="Nibali, Aiden and He, Zhen and Wollersheim, Dennis",
  title="Pulmonary nodule classification with deep residual networks",
  journal="International Journal of Computer Assisted Radiology and Surgery",
  year="2017",
  issn="1861-6429",
  doi="10.1007/s11548-017-1605-6",
}
```

And, of course, don't forget to cite the original sources of data and
annotations:

* Smith K, Clark K, Bennett W, Nolan T, Kirby J, Wolfsberger M, Moulton J,
  Vendt B, Freymann J. Data from LIDC-IDRI.
  http://dx.doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
* A. P. Reeves, A. M. Biancardi, "The Lung Image Database Consortium (LIDC)
  Nodule Size Report." Release: 2011-10-27-2. At:
  http://www.via.cornell.edu/lidc/, October 27, 2011
