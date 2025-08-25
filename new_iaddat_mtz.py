import numpy as np
import pandas as pd
import gemmi



def map_threshold(realmap, threshold, cell, negative=False):
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

def IADDAT_peaks_table(input_PDB_filename, input_MTZ_filename, input_column_labels="FoFo,PHFc", threshold_value=3.0, threshold_type="sigma", distance_cutoff=1.2):
    """
    Tabulate absolute difference density at a defined threshold within a defined cutoff distance of a model.
    Results will be output on a per-peak basis. 'Marks' are locations - there can be multiple copies of
    the model within a certain cutoff distance of symmetry related copies of density peak...this table
    stores all of them within the ASU and the integrator accounts for redundancies.

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

    Returns
    -------
    pd.DataFrame
        DataFrame with records for each peak within the distance cutoff of the model
    """
    input_PDB = gemmi.read_structure(input_PDB_filename)
    input_PDB.remove_hydrogens()
    input_PDB.remove_empty_chains()
    cell = input_PDB.cell
    model = input_PDB[0]
    input_MTZ = gemmi.read_mtz_file(input_MTZ_filename)
    # if no_water:
    #     input_PDB.remove_ligands_and_waters()
    # else:
    #     pass
    try:
        column_labels = [input_column_labels.split(',')[0], input_column_labels.split(',')[1]]
        mtz_fphi = input_MTZ.get_f_phi(column_labels[0], column_labels[1])
    except:
        print("error: please provide column labels corresponding to structure factors and phases; eg: 'FoFo,PHFc'")
        print("       columns in file: {}".format(input_MTZ.column_labels()))
        exit()
    sf  = mtz_fphi.transform_f_phi_to_map(sample_rate=4)
    asu_map = sf.masked_asu()
    realmap = np.array(asu_map.grid, copy=False)
    # sf.set_extent(input_PDB.calculate_fractional_box(margin=5)) ### alt strategy, failed upon first test on sim data
    # realmap = np.array(sf, copy=False)
    if threshold_type == "sigma":
        sites_pos = map_threshold(realmap, realmap.std()*threshold_value, input_MTZ.cell, False)
        sites_neg = map_threshold(realmap*-1, realmap.std()*threshold_value, input_MTZ.cell, True)
    elif threshold_type == "absolute":
        sites_pos = map_threshold(realmap, threshold_value, input_MTZ.cell, False)
        sites_neg = map_threshold(realmap*-1, threshold_value, input_MTZ.cell, True)
    else:
        print("error: please select valid threshold_type: 'sigma' or 'absolute'")
        exit()
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
                "mol_COM" :    COM.tolist(),
                "deltax"  :    peak.x - mark.pos.x, ### vector points toward blob
                "deltay"  :    peak.y - mark.pos.y, ### vector points toward blob
                "deltaz"  :    peak.z - mark.pos.z, ### vector points toward blob
            }
            peaks.append(record)
            

    out = pd.DataFrame.from_records(peaks)
    return out



def IADDAT_integrator(IADDAT_peaks_df, input_PDB_filename, input_MTZ_filename, threshold_value=3.0, threshold_type="sigma", distance_cutoff=1.2):
    pdb_string = input_PDB_filename.split("/")[-1].replace('.pdb','')
    mtz_string = input_MTZ_filename.split("/")[-1].replace('.mtz','')
    output_pdb_string = pdb_string+"_"+mtz_string+"_total-IADDAT-in-B-iso-{}-{}".format(str(threshold_value), threshold_type)+"_within-{}-angstroms".format(str(distance_cutoff))+".pdb"
    df_excel_string = pdb_string+"_"+mtz_string+"_IADDAT-table-{}-{}".format(str(threshold_value), threshold_type)+"_within-{}-angstroms".format(str(distance_cutoff))+".xlsx"
    input_PDB = gemmi.read_structure(input_PDB_filename)
    input_PDB.remove_hydrogens()
    input_PDB.remove_empty_chains()
    model = input_PDB[0]
    IADDAT = []
    for chain in model:
        for residue in chain:
            per_resi_IADDAT = []
            for atom in residue:
                try:
                    peaks = IADDAT_peaks_df.query('coordx=={} and coordy=={} and coordz=={}'.format(atom.pos.x, atom.pos.y, atom.pos.z))
                    unique = np.unique(peaks['peak'])
                    total_IADDAT = np.abs(unique).sum()
                except:
                    total_IADDAT = 0
                per_resi_IADDAT.append(total_IADDAT)
                atom.b_iso = total_IADDAT
                if atom.has_altloc():
                    IADDAT.append([chain.name, str(residue.seqid), residue.name, atom.name, atom.altloc, total_IADDAT])
                else:
                    IADDAT.append([chain.name, str(residue.seqid), residue.name, atom.name, "None", total_IADDAT])
            per_resi_avg_IADDAT = np.array(per_resi_IADDAT).mean()
            for atom in residue:
                atom.occ = per_resi_avg_IADDAT
    input_PDB.write_minimal_pdb(output_pdb_string)
    iaddat_df = pd.DataFrame(IADDAT, columns=["chain","residue_number","residue_name","atom_name", "atom_altloc","IADDAT"])
    iaddat_df.to_excel(df_excel_string)
    if iaddat_df['IADDAT'].sum() == 0:
        print("Warning: all IADDAT Values are 0 - inspect data and reconsider threshold_value, distance_cutoff, and ensure model matches map")
    return



def main():

    import argparse

    parser=argparse.ArgumentParser(
        description='''Integrate difference density at (and beyond; e.g.: >=3.0 & <=-3.0) a defined threshold
        within a defined cutoff distance of a model. Results will be output on a per-atom basis.''',
        epilog=""" """)
    parser.add_argument('pdb_file', type=str, help="""Standard format for molecular models""")
    parser.add_argument('mtz_file', type=str, help="""Standard format for molecular data storage - note that input columns are currently hard-coded as 'FoFo, PHFc'""")
    parser.add_argument('--threshold_value', type=float, default=3.0, help="""float (default=3.0)- level at which the map will be integrated""")
    parser.add_argument('--threshold_type', type=str, default='sigma', choices=['sigma', 'absolute'], help="""str (default='sigma')- std dev (sigma) or e-/A^3 (absolute) based threshold""")
    parser.add_argument('--distance_cutoff', type=float, default=1.2, help="""float (default=1.2)- Distance from model in angstroms at which the map will be integrated""")
    parser.add_argument('--column_labels', default="FoFo,PHFc", type=str, help="""str (default='FoFo, PHFc')- Set labels for difference structure factors and phi values""")
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
            print("error: Must provide column labels for FoFo and PhiF in a single comma-separated string; e.g.: 'FoFo,PHFc'")
            parser.print_help()
            exit(1)
    print("Integrating {} using {} at {} {} within {} angstroms of {}".format(args.mtz_file, column_labels, args.threshold_value, args.threshold_type, args.distance_cutoff, args.pdb_file))
    iaddat_df = IADDAT_peaks_table(args.pdb_file, args.mtz_file, args.column_labels, args.threshold_value, args.threshold_type, args.distance_cutoff)
    pdb_string = str(args.pdb_file).split("/")[-1].replace('.pdb','')
    mtz_string = str(args.mtz_file).split("/")[-1].replace('.mtz','')
    output_excel_string = pdb_string+"_"+mtz_string+"_peaks-table-{}-{}".format(str(args.threshold_value), args.threshold_type)+"_within-{}-angstroms".format(str(args.distance_cutoff))+".xlsx"
    iaddat_df.to_excel(output_excel_string)
    IADDAT_integrator(iaddat_df, args.pdb_file, args.mtz_file, args.threshold_value, args.threshold_type, args.distance_cutoff)
    return

if __name__ == "__main__":
    main()