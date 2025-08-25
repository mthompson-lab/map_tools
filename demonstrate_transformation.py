#!/usr/bin/env python3
"""
Demonstration script showing the difference between old and new approaches.

This script compares the old multiple-model approach with the new single-model 
inverse transformation approach for visualizing symmetry-equivalent features.
"""

import numpy as np
import pandas as pd
import gemmi

def demonstrate_transformation_approaches():
    """Demonstrate the difference between old and new approaches"""
    print("🔬 IADDAT SYMMETRY VISUALIZATION TRANSFORMATION")
    print("=" * 60)
    print("Comparing OLD vs NEW approaches for symmetry feature visualization")
    print()
    
    # Load test structure
    pdb_file = '/home/runner/work/map_tools/map_tools/test_files/3k0n.pdb'
    structure = gemmi.read_structure(pdb_file)
    
    print(f"📁 Test structure: {pdb_file}")
    print(f"   Space group: {structure.spacegroup_hm}")
    print(f"   Unit cell: {structure.cell}")
    print()
    
    # Create example peaks from different symmetry operations
    example_peaks = pd.DataFrame([
        {
            'chain': 'A', 'seqid': 100, 'residue': 'ALA', 'atom': 'CA',
            'coordx': 10.0, 'coordy': 15.0, 'coordz': 20.0,
            'peakx': 11.0, 'peaky': 16.0, 'peakz': 21.0,
            'markx': 10.5, 'marky': 15.5, 'markz': 20.5,
            'image_idx': 0, 'peak': 5.2
        },
        {
            'chain': 'A', 'seqid': 100, 'residue': 'ALA', 'atom': 'CA',
            'coordx': -10.0, 'coordy': -15.0, 'coordz': 20.0,
            'peakx': -11.0, 'peaky': -16.0, 'peakz': 21.0,
            'markx': -10.5, 'marky': -15.5, 'markz': 20.5,
            'image_idx': 1, 'peak': 4.8
        },
        {
            'chain': 'B', 'seqid': 200, 'residue': 'GLY', 'atom': 'CA',
            'coordx': -10.0, 'coordy': 15.0, 'coordz': -20.0,
            'peakx': -11.0, 'peaky': 16.0, 'peakz': -21.0,
            'markx': -10.5, 'marky': 15.5, 'markz': -20.5,
            'image_idx': 2, 'peak': -3.5
        }
    ])
    
    print(f"📊 Example data: {len(example_peaks)} peaks from {len(example_peaks['image_idx'].unique())} symmetry operations")
    print()
    
    # OLD APPROACH
    print("🔴 OLD APPROACH - Multiple Model Copies:")
    print("-" * 45)
    unique_ops = example_peaks['image_idx'].unique()
    print(f"   ❌ Creates {len(unique_ops)} model objects in PyMOL:")
    print(f"      • Original model: 'structure'")
    for i, op in enumerate(unique_ops):
        if op != 0:  # Skip identity
            print(f"      • Symmetry copy {i}: 'structure_sym_{op}'")
    
    print(f"   ❌ Each peak displayed in its symmetry-transformed space")
    print(f"   ❌ Displacement vectors created as pseudoatoms + distance lines")
    print(f"   ❌ Cluttered visualization with redundant models")
    print(f"   ❌ Higher memory usage and slower rendering")
    print()
    
    # NEW APPROACH
    print("🟢 NEW APPROACH - Inverse Transformation to Single Model:")
    print("-" * 55)
    print(f"   ✅ Uses only 1 model object: 'structure'")
    print(f"   ✅ Maps all {len(example_peaks)} peaks to original coordinate space using inverse transformations")
    print(f"   ✅ Creates proper CGO arrows for displacement vectors")
    print(f"   ✅ Clean, unified visualization on single model")
    print(f"   ✅ Better performance and memory efficiency")
    print()
    
    # MATHEMATICAL DEMONSTRATION
    print("🧮 MATHEMATICAL TRANSFORMATION EXAMPLE:")
    print("-" * 45)
    
    # Get space group operations
    sg = gemmi.SpaceGroup(structure.spacegroup_hm)
    operations = list(sg.operations())
    
    if len(operations) > 1:
        # Use the second operation as an example
        op = operations[1]
        
        # Create transformation matrix
        rot_matrix = np.array(op.rot) / 24.0
        trans_vector_frac = np.array(op.tran) / 24.0
        cell = structure.cell
        trans_vector_cart = trans_vector_frac * np.array([cell.a, cell.b, cell.c])
        
        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = rot_matrix
        transform_4x4[:3, 3] = trans_vector_cart
        
        # Example coordinate
        example_peak = example_peaks.iloc[1]  # From image_idx = 1
        symmetry_coords = np.array([example_peak['peakx'], example_peak['peaky'], example_peak['peakz'], 1.0])
        
        # Forward transformation (old approach - how we got to symmetry space)
        # This is what the old approach would apply to the original model
        print(f"   Forward transformation matrix (image_idx=1):")
        print(f"   {transform_4x4}")
        print()
        
        # Inverse transformation (new approach - mapping back to original space)
        inverse_transform = np.linalg.inv(transform_4x4)
        original_coords = inverse_transform @ symmetry_coords
        
        print(f"   Inverse transformation (NEW approach):")
        print(f"   Symmetry peak: ({symmetry_coords[0]:.1f}, {symmetry_coords[1]:.1f}, {symmetry_coords[2]:.1f})")
        print(f"   Mapped to original space: ({original_coords[0]:.1f}, {original_coords[1]:.1f}, {original_coords[2]:.1f})")
        print()
    
    # BENEFITS SUMMARY
    print("💡 KEY BENEFITS OF NEW APPROACH:")
    print("-" * 35)
    print("   🎯 Single Model Display:")
    print("      • All features visualized on one model")
    print("      • Eliminates visual clutter from model duplication")
    print("      • Easier navigation and analysis")
    print()
    print("   🚀 Performance Improvements:")
    print("      • Reduced memory usage (no model copies)")
    print("      • Faster rendering and manipulation")
    print("      • More responsive PyMOL session")
    print()
    print("   🎨 Enhanced Visualization:")
    print("      • Proper CGO arrows instead of pseudoatom lines")
    print("      • Better arrow geometry and appearance")
    print("      • Configurable arrow scaling and coloring")
    print()
    print("   🔧 Technical Advantages:")
    print("      • Mathematically correct inverse transformations")
    print("      • Preserved crystallographic accuracy")
    print("      • Modular, testable code structure")
    print()
    print("✅ TRANSFORMATION COMPLETE: From multiple models to unified visualization!")

if __name__ == "__main__":
    demonstrate_transformation_approaches()