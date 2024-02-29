# map_tools
library of map tools - beginning with IADDAT and expanding

## example from iaddat_mtz.py
```
usage: python iaddat_mtz.py [-h] [--threshold_value THRESHOLD_VALUE] [--threshold_type {sigma,absolute}] [--distance_cutoff DISTANCE_CUTOFF] [--column_labels COLUMN_LABELS] pdb_file mtz_file

Integrate difference density at (and beyond; e.g.: >=3.0 & <=-3.0) a defined threshold within a defined cutoff distance of a model. Results will be output on a per-atom basis.

positional arguments:
  pdb_file              Standard format for molecular models
  mtz_file              Standard format for molecular data storage - note that input columns are currently hard-coded as 'FoFo, PHFc'

optional arguments:
  -h, --help            show this help message and exit
  --threshold_value THRESHOLD_VALUE
                        float (default=3.0)- level at which the map will be integrated
  --threshold_type {sigma,absolute}
                        str (default='sigma')- std dev (sigma) or e-/A^3 (absolute) based threshold
  --distance_cutoff DISTANCE_CUTOFF
                        float (default=1.2)- Distance from model in angstroms at which the map will be integrated
  --column_labels COLUMN_LABELS
                        str (default='FoFo, PHFc')- Set labels for difference structure factors and phi values
```
