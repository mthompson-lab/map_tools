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
        pos = cell.orthogonalize(gemmi.Fractional(*pf))
        d  = {"x": pos.x, "y": pos.y, "z": pos.z}
        d["height"] = realmap[p[0], p[1], p[2]]
        data.append(d)
    return pd.DataFrame(data)


def IADDAT(input_PDB_filename, input_CCP4_filename, threshold_value=3, distance_cutoff=1.2):
    """
    Integrate absolute difference density at a defined threshold within a defined cutoff distance of a model.
    Results will be output on a per-atom basis.

    Parameters
    ----------
    input_PDB_filename : filepath/filename.pdb
        Standard format for molecular models
    input_CCP4_filename : filepath/filename.ccp4
        Common format for molecular map storage - note that code assumes sigma-scaled maps
    threshold_value : float
        Sigma level at which the map will be integrated
    distance_cutoff : float
        Distance from model in angstroms at which the map will be integrated

    Returns
    -------
    pd.DataFrame
        DataFrame with chain, res#, resName, and IADDAT value for each residue
    """
    input_PDB = gemmi.read_structure(input_PDB_filename)
    # if no_water:
    #     input_PDB.remove_ligands_and_waters()
    # else:
    #     pass
    input_PDB.remove_hydrogens()
    input_PDB.remove_empty_chains()
    input_map = gemmi.read_ccp4_map(input_CCP4_filename, setup=True)
    # input_map.set_extent(input_PDB.calculate_fractional_box(margin=5)) ### alt strategy, failed upon first test on sim data
    # realmap = np.array(input_map.grid, copy=False)
    asu_map = input_map.grid.masked_asu()
    realmap = np.array(asu_map.grid, copy=False)
    sites = find_sites(realmap, threshold_value, input_PDB.cell)
    sites_neg = find_sites(realmap*-1, threshold_value, input_PDB.cell)
    sites_coords = np.array([sites.x, sites.y, sites.z])
    sites_neg_coords = np.array([sites_neg.x, sites_neg.y, sites_neg.z])
    IADDAT = []
    model = input_PDB[0]
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom_coords = []
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
                # print([chain.name, str(residue.seqid), residue.name, atom.name, atom.altloc, int_pos_value, int_neg_value])
                if atom.has_altloc():
                    IADDAT.append([chain.name, str(residue.seqid), residue.name, atom.name, atom.altloc, int_pos_value, int_neg_value])
                else:
                    IADDAT.append([chain.name, str(residue.seqid), residue.name, atom.name, "None", int_pos_value, int_neg_value])
    
    return pd.DataFrame(IADDAT, columns=["chain","residue_number","residue_name","atom_name", "atom_altloc","I(+)DDAT","I(-)DDAT"])




def main():

    import argparse

    parser=argparse.ArgumentParser(
        description='''Integrate difference density at (and beyond; e.g.: >=3.0 & <=-3.0) a defined threshold
        within a defined cutoff distance of a model. Results will be output
        as an average value on a per-residue basis.''',
        epilog=""" """)
    parser.add_argument('pdb_file', type=str, help="""Standard format for molecular models""")
    parser.add_argument('ccp4_file', type=str, help="""Common format for molecular map storage - note that code assumes sigma-scaled maps""")
    parser.add_argument('--threshold_value', type=float, default=3.0, help="""float (default=3.0)- Sigma level at which the map will be integrated""")
    parser.add_argument('--distance_cutoff', type=float, default=1.2, help="""float (default=1.2)- Distance from model in angstroms at which the map will be integrated""")
    args=parser.parse_args()

    if not args.pdb_file:
        print("error: Must provide PDB file for integration")
        parser.print_help()
        exit(1)
    if not args.ccp4_file:
        print("error: Must provide CCP4 file for integration")
        parser.print_help()
        exit(1)
    print("Integrating {} at {} sigma within {} angstroms of {}".format(args.ccp4_file, args.threshold_value, args.distance_cutoff, args.pdb_file))
    iaddat_df = IADDAT(args.pdb_file, args.ccp4_file, args.threshold_value, args.distance_cutoff)
    pdb_string = str(args.pdb_file).split("/")[-1].replace('.pdb','')
    ccp4_string = str(args.ccp4_file).split("/")[-1].replace('.ccp4','')
    output_excel_string = pdb_string+"_"+ccp4_string+"_integrated-{}-sigma".format(str(args.threshold_value))+"_within-{}-angstroms".format(str(args.distance_cutoff))+".xlsx"
    iaddat_df.to_excel(output_excel_string)
    return

if __name__ == "__main__":
    main()






