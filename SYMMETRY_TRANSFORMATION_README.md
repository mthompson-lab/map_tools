# IADDAT Symmetry Visualization Transformation

## Overview

This update transforms the IADDAT plugin's approach to visualizing symmetry-equivalent peaks and displacement vectors. Instead of creating multiple model copies, the new approach uses inverse crystallographic transformations to map all symmetry features back to the original model coordinate space.

## Key Changes

### 1. Single Model Visualization
- **Before**: Created multiple PyMOL model objects for each symmetry operation
- **After**: Maps all features to a single model using inverse transformations

### 2. Inverse Transformation Mathematics
- Uses `numpy.linalg.inv()` to compute inverse transformation matrices
- Maps peaks from symmetry space back to original coordinate system
- Preserves all crystallographic accuracy while eliminating visual clutter

### 3. CGO Arrow Implementation
- **Before**: Displacement vectors shown as pseudoatoms connected by distance lines
- **After**: Proper CGO arrows with cylinder shafts and cone arrowheads
- Better visual representation and performance

## New Methods

### `map_symmetry_features_to_model(base_obj, peaks_df, pdb_file)`
Replaces `create_symmetry_equivalent_models()` with coordinate transformation approach.

### `apply_inverse_transformations_to_peaks(peaks_df, transformations, structure)`
Applies inverse crystallographic transformations to map peaks from symmetry space to original space.

### `create_displacement_vectors_with_cgo(peaks_df, threshold_value, threshold_type)`
Creates displacement vectors using CGO objects instead of pseudoatoms.

### `create_cgo_arrow(start_pos, end_pos, magnitude)`
Generates CGO arrow geometry with proper cylinder shaft and cone arrowhead.

## Benefits

1. **Performance**: Eliminates memory overhead from model duplication
2. **Clarity**: Single unified view instead of cluttered multiple models
3. **Accuracy**: Mathematically correct inverse transformations
4. **Visualization**: Superior CGO arrows vs. pseudoatom lines

## Testing

Three comprehensive test scripts validate the implementation:

- `test_symmetry_mapping.py`: Tests inverse transformation accuracy
- `test_cgo_arrows.py`: Validates CGO arrow generation
- `demonstrate_transformation.py`: Shows before/after comparison

## Compatibility

The implementation includes fallback methods to maintain compatibility with existing workflows. If the new approach fails, it automatically falls back to the original multiple-model method.

## Example Usage

```python
# Old approach (deprecated)
sym_objects = self.create_symmetry_equivalent_models(obj_name, peaks_df, pdb_file)

# New approach
mapped_peaks_df = self.map_symmetry_features_to_model(obj_name, peaks_df, pdb_file)
self.create_displacement_vectors_with_cgo(mapped_peaks_df, threshold_value, threshold_type)
```

## Mathematical Foundation

For each symmetry operation with transformation matrix T:
- **Forward**: `symmetry_coords = T @ original_coords`
- **Inverse**: `original_coords = T⁻¹ @ symmetry_coords`

The new approach uses the inverse transformation to map symmetry-equivalent features back to the original coordinate system, allowing visualization on a single model while preserving all symmetry relationships.