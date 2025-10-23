#! /bin/bash

export pdb_in=$1
mtz_out=$(sed -e "s/.pdb/_simFobs-w-Rfree.mtz/g" <<< $pdb_in)

echo "input:  ${pdb_in}"
echo "output: ${mtz_out}"

phenix.fmodel  ${pdb_in} high_resolution=0.9 label=FOBS type=real r_free=0.05 add_sigmas=True output.file_name=${mtz_out} 
