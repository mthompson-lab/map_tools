#! /bin/bash

export mtz_in=$1
mtz_out=$(sed -e "s/.mtz/_asF.mtz/g" <<< $mtz_in)

echo "input:  ${mtz_in}"
echo "output: ${mtz_out}"

phenix.reflection_file_converter  --write_mtz_amplitudes --mtz_root_label=Fobs --label=Iobs --r_free_label=FreeR_flag ${mtz_in} --mtz=${mtz_out}
