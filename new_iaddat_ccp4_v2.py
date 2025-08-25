

import numpy as np
import pandas as pd
import gemmi




def find_sites(realmap, threshold, cell, negative=False):
    """
    Find every grid point above threshold in map.

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
        if negative:
            d["height"] = realmap[p[0], p[1], p[2]]*-1
        else:
            d["height"] = realmap[p[0], p[1], p[2]]
        data.append(d)
    return pd.DataFrame(data)


def IADDAT(input_PDB_filename, input_CCP4_filename, threshold_value=3, distance_cutoff=2.0):
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
    input_PDB.remove_hydrogens()
    input_PDB.remove_empty_chains()
    cell = input_PDB.cell
    model = input_PDB[0]
    input_map = gemmi.read_ccp4_map(input_CCP4_filename, setup=True)
#     sigma_cutoff = 4
#     input_map = gemmi.read_ccp4_map(input_CCP4_filename, setup=True)
#     mean,sigma = np.mean(input_map.grid),np.std(input_map.grid)
#     cutoff = mean + sigma_cutoff * sigma
    asu_map = input_map.grid.masked_asu()
    realmap = np.array(asu_map.grid, copy=False)
    sites_pos = find_sites(realmap, threshold_value, input_PDB.cell, False)
    sites_neg = find_sites(realmap*-1, threshold_value, input_PDB.cell, True)
    sites_all = pd.concat([sites_pos,sites_neg])
    ns = gemmi.NeighborSearch(model, cell, distance_cutoff).populate()
    peaks = []
    for idx, peak in sites_all.iterrows():
        blob = gemmi.Position(peak.x,peak.y,peak.z)
        marks = ns.find_atoms(blob)
        if len(marks) == 0:
            continue

        cra = dist = None
        for mark in marks:
            image_idx = mark.image_idx
            cra = mark.to_cra(model)
            dist = cell.find_nearest_pbc_image(blob, cra.atom.pos, mark.image_idx).dist()
            COM = cra.chain.calculate_center_of_mass()
            
            if cra.atom.has_altloc():
                temp_altloc = cra.atom.altloc
            else:
                temp_altloc = None
            record = {
                "chain"   :    cra.chain.name,
                "seqid"   :    cra.residue.seqid.num,
                "residue" :    cra.residue.name,
                "atom"    :    cra.atom.name,
                "altloc"  :    temp_altloc,
                "dist"    :    dist,
                "peak"    :    peak.height,
                "coordx"  :    cra.atom.pos.x,
                "coordy"  :    cra.atom.pos.y,
                "coordz"  :    cra.atom.pos.z,
                "peakx"   :    peak.x,
                "peaky"   :    peak.y,
                "peakz"   :    peak.z,
                "markx"   :    mark.pos.x,
                "marky"   :    mark.pos.y,
                "markz"   :    mark.pos.z,
                "image_idx":   mark.image_idx,
                "mol_COM" :    COM.tolist(),
                "deltax"  :    peak.x - mark.pos.x, ### vector points toward blob
                "deltay"  :    peak.y - mark.pos.y, ### vector points toward blob
                "deltaz"  :    peak.z - mark.pos.z, ### vector points toward blob
            }
            peaks.append(record)
            

    out = pd.DataFrame.from_records(peaks)
    return out


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
