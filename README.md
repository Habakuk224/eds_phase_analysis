# Clustering-based analysis of EDS maps from EDAX APEX datasets

This is a tool which:

1. extracts data from EDAX APEX datasets
2. does a user-tunable cluster analysis for grouping the pixels into phases
3. exports the per-phase spectra into *.msa format (to be opened in DTSA analysis tool)
4. does a basic quantification of the image.

It uses Non-negative Matrix Factorization for signal decomposition and HDBSCAN for clustering.

## Usage:

`eds_phanal.py [-h] filename h5path [-a | -m] [-e ELEMENTS [ELEMENTS ...]] [-b BINNING] [-q]`

### Positional arguments:

  `filename`         Path to the data file

  `h5path`           Path within the H5 file to the map (`--map`) or to the group of maps (`--atlas`)

### Options:

  `-h, --help`            show this help message and exit

  `-a, --atlas`           Process all maps within a single H5 group.

  `-m, --map`             Process a single map.

  `-e ELEMENTS [ELEMENTS ...], --elements ELEMENTS [ELEMENTS ...]`
  List of chemical element symbols to be used; the list from H5 file is used if not provided expicitly.
  
`-b BINNING, --binning BINNING`
Spatial binning

  `-q, --quiet`           Does not open GUI, uses previously saved parameters from processing.

## GUI

