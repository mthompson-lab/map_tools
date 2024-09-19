"""Note: this doesn't perform how I'd like it to. Map gridding appears to be the issue...otherwise
blob size is the issue from flood fill. Density gets acribed to one major peak and assigned to
atoms within the distance cutoff - which is very similar to, but not the same as sampling density
at every grid point and then comparing to distance cutoffs...it artificially concentrates the
signal onto atoms."""


import numpy as np
import pandas as pd
import gemmi

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
    blobs_pos = gemmi.find_blobs_by_flood_fill(input_map.grid, cutoff=threshold_value, min_volume=0, min_score=0, 
            min_peak=0, negate=False)
    blobs_neg = gemmi.find_blobs_by_flood_fill(input_map.grid, cutoff=threshold_value, min_volume=0, min_score=0, 
            min_peak=0, negate=True)
    ns = gemmi.NeighborSearch(model, cell, distance_cutoff).populate()
    peaks = []
    for blob in blobs_pos:
        marks = ns.find_atoms(blob.centroid)
        if len(marks) == 0:
            continue

        cra = dist = None
        for mark in marks:
            image_idx = mark.image_idx
            cra = mark.to_cra(model)
            dist = cell.find_nearest_pbc_image(blob.centroid, cra.atom.pos, mark.image_idx).dist()
            record = {
                "chain"   :    cra.chain.name,
                "seqid"   :    cra.residue.seqid.num,
                "residue" :    cra.residue.name,
                "atom"    :    cra.atom.name,
                "element" :    cra.atom.element.name,
                "dist"    :    dist,
                "peak"    :    blob.peak_value,
                "cenx"    :    blob.centroid.x,
                "ceny"    :    blob.centroid.y,
                "cenz"    :    blob.centroid.z,
                "coordx"  :    cra.atom.pos.x,
                "coordy"  :    cra.atom.pos.y,
                "coordz"  :    cra.atom.pos.z,
                "deltax"  :    blob.centroid.x - mark.pos.x, ### vector points toward blob
                "deltay"  :    blob.centroid.y - mark.pos.y, ### vector points toward blob
                "deltaz"  :    blob.centroid.z - mark.pos.z, ### vector points toward blob
                "coordshiftedx" : cra.atom.pos.x + (blob.centroid.x - mark.pos.x),
                "coordshiftedy" : cra.atom.pos.y + (blob.centroid.y - mark.pos.y),
                "coordshiftedz" : cra.atom.pos.z + (blob.centroid.z - mark.pos.z),
            }
            peaks.append(record)
            
    for blob in blobs_neg:
        marks = ns.find_atoms(blob.centroid)
        if len(marks) == 0:
            continue

        cra = dist = None
        for mark in marks:
            image_idx = mark.image_idx
            cra = mark.to_cra(model)
            dist = cell.find_nearest_pbc_image(blob.centroid, cra.atom.pos, mark.image_idx).dist()
            record = {
                "chain"   :    cra.chain.name,
                "seqid"   :    cra.residue.seqid.num,
                "residue" :    cra.residue.name,
                "atom"    :    cra.atom.name,
                "element" :    cra.atom.element.name,
                "dist"    :    dist,
                "peak"    :    blob.peak_value * -1,
                "cenx"    :    blob.centroid.x,
                "ceny"    :    blob.centroid.y,
                "cenz"    :    blob.centroid.x,
                "coordx"  :    cra.atom.pos.x,
                "coordy"  :    cra.atom.pos.y,
                "coordz"  :    cra.atom.pos.z,
                "deltax"  :    mark.pos.x - blob.centroid.x, ### vector points away from blob
                "deltay"  :    mark.pos.y - blob.centroid.y, ### vector points away from blob
                "deltaz"  :    mark.pos.z - blob.centroid.z, ### vector points away from blob
                "coordshiftedx" : cra.atom.pos.x + (mark.pos.x - blob.centroid.x),
                "coordshiftedy" : cra.atom.pos.y + (mark.pos.y - blob.centroid.y),
                "coordshiftedz" : cra.atom.pos.z + (mark.pos.z - blob.centroid.z),
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
