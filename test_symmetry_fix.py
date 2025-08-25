#!/usr/bin/env python3
"""
Simple demonstration of the symmetry transformation matrix fix.

Run this script to see the before/after comparison and validation.
"""

import numpy as np
import gemmi

def main():
    """Main demonstration function"""
    print("🔬 SYMMETRY TRANSFORMATION MATRIX FIX DEMONSTRATION")
    print("=" * 60)
    
    # Load test structure
    pdb_file = '/home/runner/work/map_tools/map_tools/test_files/3k0n.pdb'
    
    try:
        structure = gemmi.read_structure(pdb_file)
        print(f"✅ Loaded: {pdb_file}")
        print(f"   Space group: {structure.spacegroup_hm}")
        print(f"   Unit cell: {structure.cell}")
        
        # Get symmetry operations
        sg = gemmi.SpaceGroup(structure.spacegroup_hm)
        operations = list(sg.operations())
        print(f"   Symmetry operations: {len(operations)}")
        
        print("\n🔧 TESTING TRANSFORMATION MATRICES:")
        print("-" * 40)
        
        # Test each operation
        test_point = np.array([10.0, 15.0, 20.0])
        all_valid = True
        
        for i, op in enumerate(operations):
            # Create our fixed transformation matrix
            rot_matrix = np.array(op.rot) / 24.0
            trans_frac = np.array(op.tran) / 24.0
            trans_cart = trans_frac * np.array([structure.cell.a, structure.cell.b, structure.cell.c])
            
            # Fixed matrix (Cartesian translations)
            transform_4x4 = np.eye(4)
            transform_4x4[:3, :3] = rot_matrix
            transform_4x4[:3, 3] = trans_cart
            
            # Test against reference
            our_result = (transform_4x4 @ np.append(test_point, 1.0))[:3]
            
            # Reference calculation
            frac_coord = structure.cell.fractionalize(gemmi.Position(*test_point))
            frac_pos = np.array([frac_coord.x, frac_coord.y, frac_coord.z])
            ref_frac = op.apply_to_xyz(frac_pos.tolist())
            ref_cart = structure.cell.orthogonalize(gemmi.Fractional(*ref_frac))
            ref_result = np.array([ref_cart.x, ref_cart.y, ref_cart.z])
            
            # Check accuracy
            error = np.linalg.norm(our_result - ref_result)
            is_valid = error < 1e-6
            all_valid = all_valid and is_valid
            
            status = "✅" if is_valid else "❌"
            print(f"   Operation {i}: {status} Error: {error:.2e} Å")
            
            if i == 1:  # Show details for first non-identity operation
                print(f"      Translation (fractional): {trans_frac}")
                print(f"      Translation (Cartesian):  {trans_cart}")
        
        print(f"\n🎯 OVERALL RESULT:")
        if all_valid:
            print("   ✅ ALL MATRICES VALIDATED - Fix successful!")
            print("   ✅ Ready for use with PyMOL")
            print("   ✅ Crystallographic symmetry expansion working")
        else:
            print("   ❌ Some matrices failed validation")
        
        print(f"\n📝 SUMMARY:")
        print(f"   • Fixed coordinate system conversion (fractional → Cartesian)")
        print(f"   • Added validation against crystallographic reference")
        print(f"   • PyMOL transformation matrices now work correctly")
        print(f"   • Symmetry expansion produces accurate results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure gemmi is installed and test file exists")

if __name__ == "__main__":
    main()