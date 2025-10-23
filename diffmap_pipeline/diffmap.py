import reciprocalspaceship as rs
import numpy as np
import argparse
parser=argparse.ArgumentParser(
    description='''diffmap calculation: model mtz1 mtz2''',
    epilog=""" """)
parser.add_argument('--model', 
                    type=str, 
                    required=True, 
                    help="str (eg: '/path/to/') - path to model for phases ")

parser.add_argument('--ground_state_mtz', 
                    type=str, 
                    required=True, 
                    help="str (eg: '/path/to/') - path to mtz1 for phases ")

parser.add_argument('--excited_state_mtz', 
                    type=str, 
                    default='', 
                    help="str (eg: '/path/to/') - path to mtz2 for phases ")

parser.add_argument('--Rfree_reference_mtz', 
                    type=str, 
                    default='', 
                    help="str (eg: '/path/to/') - path to Rfree column copying")

parser.add_argument('--output_path', 
                    type=str, 
                    default='', 
                    help="str (eg: '/path/to/diffmap.mtz') - path and filename for output")


parser.add_argument('--mode', 
                    type=str, 
                    default='diffmap', 
                    help="Choose: 'diffmap' or 'extrapmap' (default=diffmap)")



def compute_weights(df, sigdf, alpha=0):
    """
    Compute weights for each structure factor based on deltaF and its uncertainty
    """
    w = (1 + (sigdf**2 / (sigdf**2).mean()) + alpha*(df**2 / (df**2).mean()))
    return w**-1

def difference_map(ground_state_mtz, excited_state_mtz, alpha, phase_pdb_mtz, output_string_mtz):
    ground = rs.read_mtz(ground_state_mtz)
    excited = rs.read_mtz(excited_state_mtz)
    diff = ground.merge(excited, left_index=True, right_index=True, suffixes=("_ground", "_excited"))
    diff["DF"] = (diff["Fobs_excited"] - diff["Fobs_ground"]).astype("SFAmplitude")
    diff["SigDF"] = np.sqrt(diff["SIGFobs_excited"]**2 + diff["SIGFobs_ground"]**2).astype("Stddev")
    diff["W"] = compute_weights(diff["DF"], diff["SigDF"], alpha)
    diff["WDF"] = (diff["W"]*diff["DF"]).astype("F")
    ref = rs.read_mtz(phase_pdb_mtz)
    diff["PHIFMODEL"] = ref.loc[diff.index, "PHIFMODEL"]
    diff.write_mtz(output_string_mtz)
    return

def extrapolated_map(ground_state_mtz, excited_state_mtz, r_free_mtz, alpha, N, phase_pdb_mtz, output_string_mtz):
    ground = rs.read_mtz(ground_state_mtz)
    excited = rs.read_mtz(excited_state_mtz)
    r_free_mtz = rs.read_mtz(r_free_mtz)
    extrap = ground.merge(excited, left_index=True, right_index=True, suffixes=("_ground", "_excited"))
    extrap["DF"] = (extrap["Fobs_excited"] - extrap["Fobs_ground"]).astype("SFAmplitude")
    extrap["SigDF"] = np.sqrt(extrap["SIGFobs_excited"]**2 + extrap["SIGFobs_ground"]**2).astype("Stddev")
    extrap["W"] = compute_weights(extrap["DF"], extrap["SigDF"], alpha)
    extrap["WDF"] = (extrap["W"]*extrap["DF"]).astype("F")
    extrap["WSigDF"] = np.sqrt(extrap["W"]**2 * extrap["SigDF"]**2).astype("Stddev")
    extrap["ExWDF"] = (extrap["Fobs_ground"] + N*extrap["WDF"]).astype("F")
    extrap["ExWSigDF"] = np.sqrt(extrap["SIGFobs_ground"]**2 + N**2 * extrap["WSigDF"]**2).astype("Stddev")
    ref = rs.read_mtz(phase_pdb_mtz)
    extrap["PHIFMODEL"] = ref.loc[extrap.index, "PHIFMODEL"]
    try:
    	extrap["FreeR_flag"] = r_free_mtz["FreeR_flag"].reindex(extrap.index, fill_value=0)
    except:
    	extrap["R-free-flags"] = r_free_mtz["R-free-flags"].reindex(extrap.index, fill_value=0)
    extrap.write_mtz(output_string_mtz)
    return

N_value = 1
alpha_value = 0.05
args=parser.parse_args()
model_phases = args.model
ground_state_mtz = args.ground_state_mtz
excited_state_mtz = args.excited_state_mtz
Rfree_mtz = args.Rfree_reference_mtz
output_path = args.output_path
mode  = args.mode

if mode == "diffmap":
	difference_map(ground_state_mtz, excited_state_mtz, alpha_value, model_phases, output_path)

elif mode == "extrapmap":
	extrapolated_map(ground_state_mtz, excited_state_mtz, Rfree_mtz, alpha_value, N_value, model_phases, output_path)

else:
	print("Error: please choose between '--mode=diffmap' or '--mode=extrapmap' ")













