import reciprocalspaceship as rs
import numpy as np
import scipy.spatial.distance as spsd
import pandas as pd
import glob
import gemmi




def find_sites(realmap, threshold, cell):
    """
    Find local peaks in map.

    Parameters
    ----------
    realmap : np.ndarray
        3D array with voxelized electron density
    threshold : float
        Minimum voxelized density to consider for peaks
    cell : gemmi.UnitCell
        Cell parameters for crystal

    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates and peak height for each site
    """
    from skimage import feature
    peaks = feature.peak_local_max(realmap, threshold_abs=threshold, exclude_border=False)
    data = []
    for p in peaks:
        pf = p/np.array(realmap.shape)
        pf_2 = pf + np.array([-1, 0, 0])
        pf_3 = pf + np.array([0, -1, 0])
        pf_4 = pf + np.array([0, 0, -1])
        pf_5 = pf + np.array([-1, -1, 0])
        pf_6 = pf + np.array([-1, 0, -1])
        pf_7 = pf + np.array([0, -1, -1])
        pf_8 = pf + np.array([-1, -1, -1])
        pos = cell.orthogonalize(gemmi.Fractional(*pf))
        pos_2 = cell.orthogonalize(gemmi.Fractional(*pf_2))
        pos_3 = cell.orthogonalize(gemmi.Fractional(*pf_3))
        pos_4 = cell.orthogonalize(gemmi.Fractional(*pf_4))
        pos_5 = cell.orthogonalize(gemmi.Fractional(*pf_5))
        pos_6 = cell.orthogonalize(gemmi.Fractional(*pf_6))
        pos_7 = cell.orthogonalize(gemmi.Fractional(*pf_7))
        pos_8 = cell.orthogonalize(gemmi.Fractional(*pf_8))
        d  = {"x": pos.x, "y": pos.y, "z": pos.z}
        d_2  = {"x": pos_2.x, "y": pos_2.y, "z": pos_2.z}
        d_3  = {"x": pos_3.x, "y": pos_3.y, "z": pos_3.z}
        d_4  = {"x": pos_4.x, "y": pos_4.y, "z": pos_4.z}
        d_5  = {"x": pos_5.x, "y": pos_5.y, "z": pos_5.z}
        d_6  = {"x": pos_6.x, "y": pos_6.y, "z": pos_6.z}
        d_7  = {"x": pos_7.x, "y": pos_7.y, "z": pos_7.z}
        d_8  = {"x": pos_8.x, "y": pos_8.y, "z": pos_8.z}
        d["height"] = realmap[p[0], p[1], p[2]]
        d_2["height"] = realmap[p[0], p[1], p[2]]
        d_3["height"] = realmap[p[0], p[1], p[2]]
        d_4["height"] = realmap[p[0], p[1], p[2]]
        d_5["height"] = realmap[p[0], p[1], p[2]]
        d_6["height"] = realmap[p[0], p[1], p[2]]
        d_7["height"] = realmap[p[0], p[1], p[2]]
        d_8["height"] = realmap[p[0], p[1], p[2]]
        data.append(d)
        data.append(d_2)
        data.append(d_3)
        data.append(d_4)
        data.append(d_5)
        data.append(d_6)
        data.append(d_7)
        data.append(d_8)
    return pd.DataFrame(data)


def IADDAT_PDB(input_pdb, input_IADDAT, output_filename):
    res_count = 0
    for chain in input_pdb[0]:
        for residue in chain:
            res_count+=1
    if res_count == len(input_IADDAT):
        n = 0
        model = input_pdb[0]
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.b_iso = input_IADDAT[n]
                n+=1
        input_pdb.write_minimal_pdb(output_filename)
    else:
        print("failed - IADDAT and residue length do not match")
    return

def IADDAT(input_PDB_filename, input_MTZ_filename, input_column_labels=["FoFo", "PHFc"], threshold_value=3.0, distance_cutoff=2.5, average_out=True):
    """
    Integrate absolute difference density at a defined threshold within a defined cutoff distance of a model.
    Results will be output as an average value on a per-residue basis.

    Parameters
    ----------
    input_PDB_filename : filepath/filename.pdb
        Standard format for molecular models
    input_MTZ_filename : filepath/filename.mtz
        Standard format for molecular data storage - note that input columns are currently hard-coded
    threshold_value : float
        Sigma level at which the map will be integrated
    distance_cutoff : float
        Distance from model in angstroms at which the map will be integrated
    average_out : bool
        If True (default) scale per residue IADDAT value according to number of atoms in residue

    Returns
    -------
    pd.DataFrame
        DataFrame with chain, res#, resName, and IADDAT value for each residue
    """
    input_PDB = gemmi.read_structure(input_PDB_filename)
    input_PDB.remove_hydrogens()
    input_MTZ = rs.read_mtz(input_MTZ_filename)
    input_MTZ.compute_dHKL(inplace=True)
    grid_sampling = 0.25
    a_sampling = int(input_MTZ.cell.a/(input_MTZ.dHKL.min()*grid_sampling))
    b_sampling = int(input_MTZ.cell.b/(input_MTZ.dHKL.min()*grid_sampling))
    c_sampling = int(input_MTZ.cell.c/(input_MTZ.dHKL.min()*grid_sampling))
    try:
        input_MTZ["sf"] = input_MTZ.to_structurefactor(input_column_labels[0],input_column_labels[1])
    except:
	    print("error: please provide column labels in MTZ; eg: 'FoFo', 'PHFc'")
	    exit()
    reciprocalgrid = input_MTZ.to_reciprocalgrid("sf", gridsize=(a_sampling, b_sampling, c_sampling))
    realmap = np.real(np.fft.fftn(reciprocalgrid))
    sites = find_sites(realmap, realmap.std()*threshold_value, input_MTZ.cell)
    sites_neg = find_sites(realmap*-1, realmap.std()*threshold_value, input_MTZ.cell)
    sites_coords = np.array([sites.x, sites.y, sites.z])
    sites_neg_coords = np.array([sites_neg.x, sites_neg.y, sites_neg.z])
    IADDAT = []
    model = input_PDB[0]
    for chain in model:
        for residue in chain:
            atom_coords = []
            for atom in residue:
                pos  = {"x": atom.pos.x, "y": atom.pos.y, "z": atom.pos.z}
                atom_coords.append(pos)
            atom_coords_df = pd.DataFrame(atom_coords)
            distances_pos = spsd.cdist(atom_coords_df, sites_coords.transpose())
            filter_pos = []
            for row in distances_pos.transpose():
                filt_pos = np.any(row <= distance_cutoff)
                filter_pos.append(filt_pos)
            filt_pos_df = pd.DataFrame(filter_pos)

            sites["filtered"] = filt_pos_df
            integrate_pos = sites.loc[sites['filtered'] == True]
            int_pos_value = integrate_pos.height.sum()

            distances_neg = spsd.cdist(atom_coords_df, sites_neg_coords.transpose())
            filter_neg = []
            for row in distances_neg.transpose():
                filt_neg = np.any(row <= distance_cutoff)
                filter_neg.append(filt_neg)
            filt_neg_df = pd.DataFrame(filter_neg)

            sites_neg["filtered"] = filt_neg_df
            integrate_neg = sites_neg.loc[sites_neg['filtered'] == True]
            int_neg_value = integrate_neg.height.sum()

            # int_value = int_pos_value + int_neg_value
            if average_out:
            	scaled_int_pos_value = int_pos_value / len(atom_coords)
            	scaled_int_neg_value = int_neg_value / len(atom_coords)
            	IADDAT.append([chain.name, str(residue.seqid), residue.name, scaled_int_pos_value, scaled_int_neg_value])
                # new_int_value = int_value / len(atom_coords)
                # IADDAT.append([chain.name, str(residue.seqid), residue.name, new_int_value])
            else:
                # IADDAT.append([chain.name, str(residue.seqid), residue.name, int_value])
                IADDAT.append([chain.name, str(residue.seqid), residue.name, int_pos_value, int_neg_value])
    
    return pd.DataFrame(IADDAT, columns=["chain","residue_number","residue_name","I(+)DDAT","I(-)DDAT"])




def main():

	# from sys import argv
	import argparse

	parser=argparse.ArgumentParser(
	    description='''Integrate difference density at (and beyond; e.g.: >=3.0 & <=-3.0) a defined threshold
	    within a defined cutoff distance of a model. Results will be output
	    as an average value on a per-residue basis.''',
	    epilog=""" """)
	parser.add_argument('pdb_file', type=str, help="""Standard format for molecular models""")
	parser.add_argument('mtz_file', type=str, help="""Standard format for molecular data storage - note that input columns are currently hard-coded as 'FoFo, PHFc'""")
	parser.add_argument('--threshold_value', type=float, default=3.0, help="""float (default=3.0)- Sigma level at which the map will be integrated""")
	parser.add_argument('--distance_cutoff', type=float, default=2.5, help="""float (default=2.5)- Distance from model in angstroms at which the map will be integrated""")
	parser.add_argument('--column_labels', default="FoFo, PHFc", type=str, help="""str (default=["FoFo", "PHFc"])- Set labels for difference structure factors and phi values""")
	parser.add_argument('--average_out', action='store_true',  help="""Default: scales avg IADDAT per residue / number of atoms in residue""")
	parser.add_argument('--no-average_out', dest='average_out', action='store_false',  help="""Turns off scaling per number of atoms in residue""")
	parser.set_defaults(average_out=True)


	args=parser.parse_args()

	if not args.pdb_file:
	    print("error: Must provide PDB file for integration")
	    parser.print_help()
	    exit(1)

	if not args.mtz_file:
	    print("error: Must provide MTZ file for integration")
	    parser.print_help()
	    exit(1)
	if args.column_labels:
		column_labels = [args.column_labels.split(',')[0], args.column_labels.split(',')[1]]
		if len(column_labels) != 2:
		    print("error: Must provide column labels for FoFo and PhiF in a single comma-separated string; e.g.: 'FoFo, PHFc'")
		    parser.print_help()
		    exit(1)
	print("Integrating {} using {} at {} sigma within {} angstroms of {}".format(args.mtz_file, column_labels, args.threshold_value, args.distance_cutoff, args.pdb_file))
	print("Per residue IADDAT values scaled by number of atoms per residue = {}".format(args.average_out))

	iaddat_df = IADDAT(args.pdb_file, args.mtz_file, column_labels, args.threshold_value, args.distance_cutoff, args.average_out)
	pdb_string = str(args.pdb_file).split("/")[-1].replace('.pdb','')
	mtz_string = str(args.mtz_file).split("/")[-1].replace('.mtz','')
	output_excel_string = pdb_string+"_"+mtz_string+"_integrated-{}-sigma".format(str(args.threshold_value))+"_within-{}-angstroms".format(str(args.distance_cutoff))+".xlsx"
	iaddat_df.to_excel(output_excel_string)
	return

if __name__ == "__main__":
    main()






