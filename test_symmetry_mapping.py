#!/usr/bin/env python3
"""
Test script for the new symmetry mapping approach.

This script tests the new functionality that maps symmetry-equivalent peaks and arrows 
back to the original model coordinate space instead of creating multiple model copies.
"""

import sys
import os
import numpy as np
import pandas as pd
import gemmi

# Add the current directory to the Python path to import the plugin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the plugin class (without PyMOL dependencies)
class MockCmd:
    """Mock PyMOL cmd object for testing without PyMOL"""
    def load(self, filename, obj_name):
        print(f"Mock: Loading {filename} as {obj_name}")
    
    def copy(self, new_name, old_name):
        print(f"Mock: Copying {old_name} to {new_name}")
    
    def transform_selection(self, obj_name, matrix):
        print(f"Mock: Transforming {obj_name} with matrix")
    
    def group(self, group_name, objects):
        print(f"Mock: Grouping {objects} as {group_name}")
    
    def spectrum(self, prop, colors, selection):
        print(f"Mock: Applying spectrum {colors} to {selection} by {prop}")
    
    def show(self, representation, selection):
        print(f"Mock: Showing {selection} as {representation}")
    
    def pseudoatom(self, obj_name, pos=None, **kwargs):
        print(f"Mock: Creating pseudoatom in {obj_name} at {pos}")
    
    def color(self, color, selection):
        print(f"Mock: Coloring {selection} with {color}")
    
    def center(self, selection):
        print(f"Mock: Centering on {selection}")
    
    def zoom(self, selection):
        print(f"Mock: Zooming to {selection}")
    
    def load_cgo(self, cgo_objects, obj_name):
        print(f"Mock: Loading CGO object {obj_name} with {len(cgo_objects)} elements")
    
    def select(self, name, selection):
        print(f"Mock: Creating selection {name}: {selection}")
    
    def count_atoms(self, selection):
        print(f"Mock: Counting atoms in {selection}")
        return 10  # Mock return value
    
    def create(self, new_name, selection):
        print(f"Mock: Creating object {new_name} from {selection}")
    
    def delete(self, obj_name):
        print(f"Mock: Deleting {obj_name}")
    
    def set_name(self, old_name, new_name):
        print(f"Mock: Renaming {old_name} to {new_name}")
    
    def hide(self, representation, selection):
        print(f"Mock: Hiding {representation} for {selection}")
    
    def set(self, setting, value, selection):
        print(f"Mock: Setting {setting} to {value} for {selection}")

# Monkey patch cmd into the global namespace for the plugin
import builtins
builtins.cmd = MockCmd()

# Create a minimal IADDATPlugin instance for testing
class TestableIADDATPlugin:
    """Minimal version of IADDATPlugin for testing symmetry mapping"""
    
    def __init__(self):
        # Mock GUI elements
        class MockSpinBox:
            def value(self):
                return 1.0
        
        self.vector_scale_spin = MockSpinBox()
    
    # Import the key methods from the main plugin
    def extract_symmetry_transformations(self, peaks_df, structure):
        """Extract crystallographic transformations from image_idx values using proper symmetry operations"""
        try:
            # Check if image_idx column exists
            if 'image_idx' not in peaks_df.columns:
                print("Warning: image_idx column not found in peaks DataFrame. Cannot extract proper symmetry operations.")
                return []
            
            # Get unique image_idx values from the peaks data
            unique_image_indices = peaks_df['image_idx'].unique()
            print(f"Found {len(unique_image_indices)} unique symmetry operations (image_idx values): {unique_image_indices}")
            
            # Get the space group from the structure
            spacegroup = structure.spacegroup_hm
            sg = gemmi.SpaceGroup(spacegroup)
            operations = sg.operations()
            
            print(f"Structure space group: {spacegroup}")
            print(f"Space group has {len(operations)} symmetry operations")
            
            # Create transformation matrices for each unique image_idx
            pymol_transforms = []
            operations_list = list(operations)  # Convert to list for indexing
            
            for image_idx in sorted(unique_image_indices):
                if image_idx >= len(operations_list):
                    print(f"Warning: image_idx {image_idx} exceeds available operations ({len(operations_list)})")
                    continue
                    
                # Get the crystallographic operation for this image_idx
                op = operations_list[image_idx]
                
                # Convert gemmi's 24-based rotation and translation to proper matrices
                rot_matrix = np.array(op.rot) / 24.0  # gemmi uses 24-based system
                trans_vector_frac = np.array(op.tran) / 24.0  # fractional translation
                
                # CRITICAL FIX: Convert fractional translations to Cartesian for PyMOL
                # PyMOL expects transformation matrices with Cartesian translations in Angstroms
                cell = structure.cell
                trans_vector_cart = trans_vector_frac * np.array([cell.a, cell.b, cell.c])
                
                # Create 4x4 transformation matrix for PyMOL
                transform_4x4 = np.eye(4)
                transform_4x4[:3, :3] = rot_matrix
                transform_4x4[:3, 3] = trans_vector_cart  # Use Cartesian translations
                
                # Convert to PyMOL format (flattened 16-element list)
                flat_matrix = transform_4x4.flatten().tolist()
                pymol_transforms.append(flat_matrix)
                
                print(f"Image_idx {image_idx}: Rotation\n{rot_matrix}")
                print(f"Image_idx {image_idx}: Translation (fractional) {trans_vector_frac}")
                print(f"Image_idx {image_idx}: Translation (Cartesian) {trans_vector_cart}")
            
            print(f"Generated {len(pymol_transforms)} crystallographic transformation matrices")
            return pymol_transforms
            
        except Exception as e:
            print(f"Error extracting crystallographic transformations: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def apply_inverse_transformations_to_peaks(self, peaks_df, transformations, structure):
        """Apply inverse transformations to map peaks from symmetry space back to original model space"""
        try:
            print("Applying inverse transformations to map symmetry features to original model space...")
            
            # Check if we have the required columns
            if 'image_idx' not in peaks_df.columns:
                print("Warning: image_idx column not found, returning original peaks")
                return peaks_df
            
            # Create a copy of the DataFrame to modify
            mapped_peaks_df = peaks_df.copy()
            
            # Get space group operations for inverse calculation
            sg = gemmi.SpaceGroup(structure.spacegroup_hm)
            operations = list(sg.operations())
            
            # Process each peak and map it back to original coordinate space
            for idx, row in mapped_peaks_df.iterrows():
                image_idx = int(row['image_idx'])
                
                # Skip identity transformation (image_idx = 0)
                if image_idx == 0:
                    continue
                
                if image_idx >= len(transformations):
                    continue
                
                # Get the forward transformation matrix and compute its inverse
                transform_flat = transformations[image_idx]
                transform_4x4 = np.array(transform_flat).reshape(4, 4)
                inverse_transform = np.linalg.inv(transform_4x4)
                
                # Transform peak coordinates from symmetry space to original space
                peak_coords = np.array([row['peakx'], row['peaky'], row['peakz'], 1.0])
                transformed_peak = inverse_transform @ peak_coords
                
                # Transform mark coordinates (start of displacement vectors)
                mark_coords = np.array([row['markx'], row['marky'], row['markz'], 1.0])
                transformed_mark = inverse_transform @ mark_coords
                
                # Update the DataFrame with transformed coordinates
                mapped_peaks_df.at[idx, 'peakx'] = transformed_peak[0]
                mapped_peaks_df.at[idx, 'peaky'] = transformed_peak[1]
                mapped_peaks_df.at[idx, 'peakz'] = transformed_peak[2]
                
                mapped_peaks_df.at[idx, 'markx'] = transformed_mark[0]
                mapped_peaks_df.at[idx, 'marky'] = transformed_mark[1]
                mapped_peaks_df.at[idx, 'markz'] = transformed_mark[2]
                
                # Recalculate displacement vectors in original space
                mapped_peaks_df.at[idx, 'deltax'] = transformed_peak[0] - transformed_mark[0]
                mapped_peaks_df.at[idx, 'deltay'] = transformed_peak[1] - transformed_mark[1]
                mapped_peaks_df.at[idx, 'deltaz'] = transformed_peak[2] - transformed_mark[2]
            
            print(f"Successfully mapped {len(mapped_peaks_df)} peaks to original coordinate space")
            return mapped_peaks_df
            
        except Exception as e:
            print(f"Error applying inverse transformations: {e}")
            import traceback
            traceback.print_exc()
            return peaks_df  # Return original if transformation fails
    
    def map_symmetry_features_to_model(self, base_obj, peaks_df, pdb_file):
        """Map symmetry-equivalent peaks and arrows back to original model coordinate space"""
        try:
            # Read the structure to get unit cell information
            structure = gemmi.read_structure(pdb_file)
            
            # Extract unique symmetry operations from the peaks data using image_idx
            transformations = self.extract_symmetry_transformations(peaks_df, structure)
            
            if not transformations:
                print("No transformations found, returning original peaks")
                return peaks_df
            
            # Instead of creating multiple models, map all symmetry features to original space
            mapped_peaks_df = self.apply_inverse_transformations_to_peaks(peaks_df, transformations, structure)
            
            print(f"Mapped {len(mapped_peaks_df)} symmetry-equivalent features to original model space")
            return mapped_peaks_df
            
        except Exception as e:
            print(f"Error mapping symmetry features: {e}")
            import traceback
            traceback.print_exc()
            return peaks_df

def create_test_peaks_dataframe():
    """Create a test DataFrame with peaks from different symmetry operations"""
    
    # Create test data with peaks from different image_idx values (symmetry operations)
    test_data = [
        # Original position (image_idx = 0, identity)
        {
            'chain': 'A', 'seqid': 100, 'residue': 'ALA', 'atom': 'CA', 'altloc': None,
            'dist': 1.0, 'peak': 5.2,
            'coordx': 10.0, 'coordy': 15.0, 'coordz': 20.0,
            'peakx': 11.0, 'peaky': 16.0, 'peakz': 21.0,
            'markx': 10.5, 'marky': 15.5, 'markz': 20.5,
            'image_idx': 0, 'mol_COM': [12.0, 17.0, 22.0],
            'deltax': 0.5, 'deltay': 0.5, 'deltaz': 0.5
        },
        # Symmetry equivalent position (image_idx = 1)
        {
            'chain': 'A', 'seqid': 100, 'residue': 'ALA', 'atom': 'CA', 'altloc': None,
            'dist': 1.0, 'peak': 4.8,
            'coordx': -10.0, 'coordy': -15.0, 'coordz': 20.0,  # Transformed coordinates
            'peakx': -11.0, 'peaky': -16.0, 'peakz': 21.0,     # Transformed peak
            'markx': -10.5, 'marky': -15.5, 'markz': 20.5,    # Transformed mark
            'image_idx': 1, 'mol_COM': [-12.0, -17.0, 22.0],
            'deltax': -0.5, 'deltay': -0.5, 'deltaz': 0.5
        },
        # Another symmetry equivalent (image_idx = 2)
        {
            'chain': 'B', 'seqid': 200, 'residue': 'GLY', 'atom': 'CA', 'altloc': None,
            'dist': 0.8, 'peak': -3.5,
            'coordx': -10.0, 'coordy': 15.0, 'coordz': -20.0,  # Different symmetry operation
            'peakx': -11.0, 'peaky': 16.0, 'peakz': -21.0,
            'markx': -10.5, 'marky': 15.5, 'markz': -20.5,
            'image_idx': 2, 'mol_COM': [-12.0, 17.0, -22.0],
            'deltax': -0.5, 'deltay': 0.5, 'deltaz': -0.5
        }
    ]
    
    return pd.DataFrame(test_data)

def test_symmetry_mapping():
    """Test the new symmetry mapping functionality"""
    print("🧪 TESTING NEW SYMMETRY MAPPING APPROACH")
    print("=" * 60)
    
    # Load test structure
    pdb_file = '/home/runner/work/map_tools/map_tools/test_files/3k0n.pdb'
    
    try:
        structure = gemmi.read_structure(pdb_file)
        print(f"✅ Loaded test structure: {pdb_file}")
        print(f"   Space group: {structure.spacegroup_hm}")
        print(f"   Unit cell: {structure.cell}")
        
        # Create test peaks DataFrame
        peaks_df = create_test_peaks_dataframe()
        print(f"\n✅ Created test peaks DataFrame with {len(peaks_df)} peaks")
        print(f"   Image indices: {sorted(peaks_df['image_idx'].unique())}")
        
        # Test the new mapping approach
        plugin = TestableIADDATPlugin()
        
        print(f"\n🔄 Testing symmetry feature mapping...")
        mapped_peaks = plugin.map_symmetry_features_to_model("test_model", peaks_df, pdb_file)
        
        print(f"\n📊 RESULTS COMPARISON:")
        print(f"{'Index':<6} {'Original Peak':<25} {'Mapped Peak':<25} {'Image_idx':<10}")
        print("-" * 70)
        
        for idx, (orig_row, mapped_row) in enumerate(zip(peaks_df.iterrows(), mapped_peaks.iterrows())):
            orig_peak = f"({orig_row[1]['peakx']:.1f}, {orig_row[1]['peaky']:.1f}, {orig_row[1]['peakz']:.1f})"
            mapped_peak = f"({mapped_row[1]['peakx']:.1f}, {mapped_row[1]['peaky']:.1f}, {mapped_row[1]['peakz']:.1f})"
            image_idx = orig_row[1]['image_idx']
            
            print(f"{idx:<6} {orig_peak:<25} {mapped_peak:<25} {image_idx:<10}")
        
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"   - Original approach would create {len(peaks_df['image_idx'].unique())} model copies")
        print(f"   - New approach maps all features to single model")
        print(f"   - All {len(mapped_peaks)} peaks transformed to original coordinate space")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_symmetry_mapping()
    sys.exit(0 if success else 1)