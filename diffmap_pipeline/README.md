# map_pipeline
Take merged structure factors and create difference or extrapolated structure factors.

### Dependencies:
* PHENIX (https://www.phenix-online.org/)
* Reciprocal Spaceship (https://rs-station.github.io/)

### Assumptions:
* Data are already on a common scale
  * MTZ files from cctbx.xfel that were scaled and merged using the same reference PDB or MTZ and merging parameters
  * MTZ files put on a common scale using SCALEIT
* Intensities are provided and will be converted to structure factors


### Example usage:

1) Gather all MTZ files and a reference PDB structure refined against the ground state structure factors.
2) Choose a set of R-free-flags to be mapped onto all data, we recommend simulating extremely high-resolution structure factors with matching R-Free-flags from the ground state model.
```
./phenix_rfreesim.sh ground_state_model.pdb
```
3) Convert intensities to structure factors for each MTZ of interest
```
./phenix_asF.sh data.mtz
```
4) Generate difference structure factors for each comparison of interest (note this code can also compute extrapolated structure factors)
```
python diffmap.py --model GROUND_STATE_PDB --ground_state_mtz GROUND_STATE_MTZ --excited_state_mtz EXCITED_STATE_MTZ --output_path OUTPUT_PATH --mode=diffmap
```