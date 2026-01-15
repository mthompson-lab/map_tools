"""
IADDAT PyMOL Plugin
Integrate Absolute Difference Density Around aToms

This plugin provides a GUI interface for the IADDAT analysis within PyMOL.
"""

import sys
import os
try:
    from pymol import cmd
    from pymol.Qt import QtWidgets, QtCore
    import pymol
except ImportError:
    print("This plugin requires PyMOL")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    import gemmi
    from skimage import feature
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: numpy, pandas, gemmi, scikit-image")

def __init_plugin__(app=None):
    """
    Add an entry to the PyMOL menu
    """
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('IADDAT Analysis', run_plugin)

class IADDATPlugin(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('IADDAT - Integrate Absolute Difference Density Around aToms')
        self.setGeometry(300, 300, 600, 400)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = QtWidgets.QLabel('IADDAT Analysis')
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QtWidgets.QLabel(
            'Integrate difference density at a defined threshold within a cutoff distance of a model.\n'
            'Results will be output on a per-atom basis with visualization in PyMOL.'
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("margin: 5px; color: #666;")
        layout.addWidget(desc)
        
        # File inputs
        files_group = QtWidgets.QGroupBox("Input Files")
        files_layout = QtWidgets.QFormLayout()
        
        # PDB file input
        pdb_layout = QtWidgets.QHBoxLayout()
        self.pdb_entry = QtWidgets.QLineEdit()
        self.pdb_entry.setPlaceholderText("Select PDB file...")
        pdb_browse = QtWidgets.QPushButton("Browse")
        pdb_browse.clicked.connect(lambda: self.browse_file(self.pdb_entry, "PDB Files (*.pdb)"))
        pdb_layout.addWidget(self.pdb_entry)
        pdb_layout.addWidget(pdb_browse)
        files_layout.addRow("PDB File:", pdb_layout)
        
        # MTZ file input
        mtz_layout = QtWidgets.QHBoxLayout()
        self.mtz_entry = QtWidgets.QLineEdit()
        self.mtz_entry.setPlaceholderText("Select MTZ file...")
        mtz_browse = QtWidgets.QPushButton("Browse")
        mtz_browse.clicked.connect(lambda: self.browse_file(self.mtz_entry, "MTZ Files (*.mtz)"))
        mtz_layout.addWidget(self.mtz_entry)
        mtz_layout.addWidget(mtz_browse)
        files_layout.addRow("MTZ File:", mtz_layout)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        # Parameters
        params_group = QtWidgets.QGroupBox("Parameters")
        params_layout = QtWidgets.QFormLayout()
        
        # Column labels
        self.column_entry = QtWidgets.QLineEdit("FoFo,PHFc")
        self.column_entry.setToolTip("Comma-separated column labels for structure factors and phases")
        params_layout.addRow("Column Labels:", self.column_entry)
        
        # Threshold value
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 20.0)
        self.threshold_spin.setValue(3.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setDecimals(2)
        params_layout.addRow("Threshold Value:", self.threshold_spin)
        
        # Threshold type
        self.threshold_combo = QtWidgets.QComboBox()
        self.threshold_combo.addItems(["sigma", "absolute"])
        params_layout.addRow("Threshold Type:", self.threshold_combo)
        
        # Distance cutoff
        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin.setRange(0.1, 10.0)
        self.distance_spin.setValue(1.2)
        self.distance_spin.setSingleStep(0.1)
        self.distance_spin.setDecimals(1)
        self.distance_spin.setSuffix(" Å")
        params_layout.addRow("Distance Cutoff:", self.distance_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Visualization options
        viz_group = QtWidgets.QGroupBox("Visualization Options")
        viz_layout = QtWidgets.QVBoxLayout()
        
        self.load_results_check = QtWidgets.QCheckBox("Load results into PyMOL")
        self.load_results_check.setChecked(True)
        viz_layout.addWidget(self.load_results_check)
        
        self.color_by_iaddat_check = QtWidgets.QCheckBox("Color atoms by IADDAT values")
        self.color_by_iaddat_check.setChecked(True)
        viz_layout.addWidget(self.color_by_iaddat_check)
        
        self.show_peaks_check = QtWidgets.QCheckBox("Show density peaks as spheres")
        self.show_peaks_check.setChecked(True)
        viz_layout.addWidget(self.show_peaks_check)
        
        self.carve_peaks_check = QtWidgets.QCheckBox("Carve peaks around model (recommended)")
        self.carve_peaks_check.setChecked(True)
        viz_layout.addWidget(self.carve_peaks_check)
        
        self.show_vectors_check = QtWidgets.QCheckBox("Show per-atom displacement vectors as arrows")
        self.show_vectors_check.setChecked(True)
        viz_layout.addWidget(self.show_vectors_check)
        
        # Vector scaling
        vector_layout = QtWidgets.QHBoxLayout()
        vector_layout.addWidget(QtWidgets.QLabel("Vector scale factor:"))
        self.vector_scale_spin = QtWidgets.QDoubleSpinBox()
        self.vector_scale_spin.setRange(0.1, 10.0)
        self.vector_scale_spin.setValue(2.0)
        self.vector_scale_spin.setSingleStep(0.1)
        self.vector_scale_spin.setDecimals(1)
        vector_layout.addWidget(self.vector_scale_spin)
        viz_layout.addLayout(vector_layout)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: green; margin: 5px;")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.run_button = QtWidgets.QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def browse_file(self, entry_widget, file_filter):
        """Browse for file and update entry widget"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if filename:
            entry_widget.setText(filename)
    
    def update_status(self, message, color="black"):
        """Update status label"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; margin: 5px;")
        QtCore.QCoreApplication.processEvents()
    
    def validate_inputs(self):
        """Validate user inputs"""
        if not self.pdb_entry.text().strip():
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a PDB file.")
            return False
        
        if not self.mtz_entry.text().strip():
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select an MTZ file.")
            return False
        
        if not os.path.exists(self.pdb_entry.text()):
            QtWidgets.QMessageBox.warning(self, "Warning", "PDB file does not exist.")
            return False
        
        if not os.path.exists(self.mtz_entry.text()):
            QtWidgets.QMessageBox.warning(self, "Warning", "MTZ file does not exist.")
            return False
        
        column_labels = self.column_entry.text().strip().split(',')
        if len(column_labels) != 2:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please provide exactly two column labels separated by comma.")
            return False
        
        return True
    
    def run_analysis(self):
        """Run the IADDAT analysis"""
        if not self.validate_inputs():
            return
        
        try:
            self.run_button.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # Indeterminate progress
            self.update_status("Running IADDAT analysis...", "blue")
            
            # Get parameters
            pdb_file = self.pdb_entry.text().strip()
            mtz_file = self.mtz_entry.text().strip()
            column_labels = self.column_entry.text().strip()
            threshold_value = self.threshold_spin.value()
            threshold_type = self.threshold_combo.currentText()
            distance_cutoff = self.distance_spin.value()
            
            # Run the analysis
            self.update_status("Generating peaks table...", "blue")
            peaks_df = IADDAT_peaks_table(
                pdb_file, mtz_file, column_labels,
                threshold_value, threshold_type, distance_cutoff
            )
            
            self.update_status("Integrating IADDAT values...", "blue")
            IADDAT_integrator(
                peaks_df, pdb_file, mtz_file,
                threshold_value, threshold_type, distance_cutoff
            )
            
            # Load results into PyMOL if requested
            if self.load_results_check.isChecked():
                self.load_results_to_pymol(pdb_file, mtz_file, peaks_df, threshold_value, threshold_type, distance_cutoff)
            
            # Generate and display vector analytics if vectors are enabled
            if self.show_vectors_check.isChecked() and not peaks_df.empty:
                self.update_status("Generating vector analytics...", "blue")
                analytics = self.generate_vector_analytics(peaks_df)
                if analytics:
                    self.display_analytics_summary(analytics)
            
            self.update_status("Analysis completed successfully!", "green")
            
            # Show completion message
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"IADDAT analysis completed!\n\n"
                f"Files generated:\n"
                f"- PDB with IADDAT B-factors\n"
                f"- Excel file with IADDAT table\n"
                f"- Excel file with peaks table"
            )
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
        
        finally:
            self.run_button.setEnabled(True)
            self.progress.setVisible(False)
    
    def load_results_to_pymol(self, pdb_file, mtz_file, peaks_df, threshold_value, threshold_type, distance_cutoff):
        """Load results into PyMOL for visualization anchored to the model"""
        try:
            # Generate output filenames (matching the original script logic)
            pdb_string = os.path.basename(pdb_file).replace('.pdb', '')
            mtz_string = os.path.basename(mtz_file).replace('.mtz', '')
            output_pdb = f"{pdb_string}_{mtz_string}_total-IADDAT-in-B-iso-{threshold_value}-{threshold_type}_within-{distance_cutoff}-angstroms.pdb"
            
            if os.path.exists(output_pdb):
                # Load the PDB with IADDAT values
                obj_name = f"IADDAT_{pdb_string}"
                try:
                    cmd.load(output_pdb, obj_name)
                except NameError:
                    print(f"Would load PDB file: {output_pdb} as object: {obj_name}")
                
                if self.color_by_iaddat_check.isChecked():
                    # Color by B-factor (which contains IADDAT values)
                    try:
                        cmd.spectrum("b", "blue_white_red", obj_name)
                        cmd.show("sticks", obj_name)
                    except NameError:
                        print(f"Would color {obj_name} by B-factor")
                
                # Show density peaks as spheres if requested (anchored to model)
                if self.show_peaks_check.isChecked() and not peaks_df.empty:
                    if self.carve_peaks_check.isChecked():
                        self.create_carved_peak_objects_single_model(peaks_df, obj_name, threshold_value, threshold_type, distance_cutoff)
                    else:
                        self.create_peak_objects_single_model(peaks_df, threshold_value, threshold_type)
                
                # Show displacement vectors if requested (using CGO arrows anchored to model)
                if self.show_vectors_check.isChecked() and not peaks_df.empty:
                    self.create_displacement_vectors_with_cgo(peaks_df, threshold_value, threshold_type)
                
                # Center view on the structure
                try:
                    cmd.center(obj_name)
                    cmd.zoom(obj_name)
                except NameError:
                    print(f"Would center and zoom on {obj_name}")
                
                self.update_status("Results loaded into PyMOL", "green")
            
        except Exception as e:
            self.update_status(f"Error loading to PyMOL: {str(e)}", "red")

    def create_peak_objects_single_model(self, peaks_df, threshold_value, threshold_type):
        """Create PyMOL objects for density peaks relative to model atoms"""
        try:
            # Create positive and negative peak objects
            pos_peaks = peaks_df[peaks_df['peak'] > 0]
            neg_peaks = peaks_df[peaks_df['peak'] < 0]
            
            if not pos_peaks.empty:
                # Create positive peaks (red spheres) at coord + delta
                pos_obj = f"positive_peaks_{threshold_value}_{threshold_type}"
                for _, peak in pos_peaks.iterrows():
                    # Peak position is original atom coord + delta
                    peak_x = peak['coordx'] + peak['deltax']
                    peak_y = peak['coordy'] + peak['deltay']
                    peak_z = peak['coordz'] + peak['deltaz']
                    try:
                        cmd.pseudoatom(pos_obj, pos=[peak_x, peak_y, peak_z], 
                                     b=peak['peak'], vdw=0.3)
                    except NameError:
                        print(f"Would create positive peak at ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f})")
                
                try:
                    cmd.color("red", pos_obj)
                    cmd.show("spheres", pos_obj)
                except NameError:
                    print(f"Would color {pos_obj} red and show as spheres")
            
            if not neg_peaks.empty:
                # Create negative peaks (blue spheres) at coord + delta
                neg_obj = f"negative_peaks_{threshold_value}_{threshold_type}"
                for _, peak in neg_peaks.iterrows():
                    # Peak position is original atom coord + delta
                    peak_x = peak['coordx'] + peak['deltax']
                    peak_y = peak['coordy'] + peak['deltay']
                    peak_z = peak['coordz'] + peak['deltaz']
                    try:
                        cmd.pseudoatom(neg_obj, pos=[peak_x, peak_y, peak_z], 
                                     b=abs(peak['peak']), vdw=0.3)
                    except NameError:
                        print(f"Would create negative peak at ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f})")
                
                try:
                    cmd.color("blue", neg_obj)
                    cmd.show("spheres", neg_obj)
                except NameError:
                    print(f"Would color {neg_obj} blue and show as spheres")
            
        except Exception as e:
            print(f"Error creating peak objects: {e}")

    def create_carved_peak_objects_single_model(self, peaks_df, model_obj, threshold_value, threshold_type, distance_cutoff):
        """Create PyMOL objects for density peaks relative to model atoms, carved around the model"""
        try:
            # Create positive and negative peak objects
            pos_peaks = peaks_df[peaks_df['peak'] > 0]
            neg_peaks = peaks_df[peaks_df['peak'] < 0]
            
            if not pos_peaks.empty:
                # Create positive peaks (red spheres) at coord + delta
                pos_obj = f"positive_peaks_carved_{threshold_value}_{threshold_type}"
                for _, peak in pos_peaks.iterrows():
                    # Peak position is original atom coord + delta
                    peak_x = peak['coordx'] + peak['deltax']
                    peak_y = peak['coordy'] + peak['deltay']
                    peak_z = peak['coordz'] + peak['deltaz']
                    try:
                        cmd.pseudoatom(pos_obj, pos=[peak_x, peak_y, peak_z], 
                                     b=peak['peak'], vdw=0.3)
                    except NameError:
                        print(f"Would create carved positive peak at ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f})")
                
                try:
                    cmd.color("red", pos_obj)
                    cmd.show("spheres", pos_obj)
                    
                    # Carve around the model
                    cmd.select(f"temp_sel_{pos_obj}", f"{pos_obj} within {distance_cutoff*1.5} of {model_obj}")
                    if cmd.count_atoms(f"temp_sel_{pos_obj}") > 0:
                        cmd.create(f"{pos_obj}_carved", f"temp_sel_{pos_obj}")
                        cmd.delete(pos_obj)
                        cmd.set_name(f"{pos_obj}_carved", pos_obj)
                    cmd.delete(f"temp_sel_{pos_obj}")
                except NameError:
                    print(f"Would carve {pos_obj} around {model_obj}")
            
            if not neg_peaks.empty:
                # Create negative peaks (blue spheres) at coord + delta
                neg_obj = f"negative_peaks_carved_{threshold_value}_{threshold_type}"
                for _, peak in neg_peaks.iterrows():
                    # Peak position is original atom coord + delta
                    peak_x = peak['coordx'] + peak['deltax']
                    peak_y = peak['coordy'] + peak['deltay']
                    peak_z = peak['coordz'] + peak['deltaz']
                    try:
                        cmd.pseudoatom(neg_obj, pos=[peak_x, peak_y, peak_z], 
                                     b=abs(peak['peak']), vdw=0.3)
                    except NameError:
                        print(f"Would create carved negative peak at ({peak_x:.2f}, {peak_y:.2f}, {peak_z:.2f})")
                
                try:
                    cmd.color("blue", neg_obj)
                    cmd.show("spheres", neg_obj)
                    
                    # Carve around the model
                    cmd.select(f"temp_sel_{neg_obj}", f"{neg_obj} within {distance_cutoff*1.5} of {model_obj}")
                    if cmd.count_atoms(f"temp_sel_{neg_obj}") > 0:
                        cmd.create(f"{neg_obj}_carved", f"temp_sel_{neg_obj}")
                        cmd.delete(neg_obj)
                        cmd.set_name(f"{neg_obj}_carved", neg_obj)
                    cmd.delete(f"temp_sel_{neg_obj}")
                except NameError:
                    print(f"Would carve {neg_obj} around {model_obj}")
            
        except Exception as e:
            print(f"Error creating carved peak objects: {e}")
    
    def create_carved_peak_objects(self, peaks_df, model_objects, threshold_value, threshold_type, distance_cutoff):
        """Create PyMOL objects for density peaks carved around all symmetry-equivalent models"""
        try:
            # Create positive and negative peak objects
            pos_peaks = peaks_df[peaks_df['peak'] > 0]
            neg_peaks = peaks_df[peaks_df['peak'] < 0]
            
            # Combine all model objects for carving selection
            if isinstance(model_objects, list):
                model_selection = " or ".join(model_objects)
            else:
                model_selection = model_objects
            
            if not pos_peaks.empty:
                # Create positive peaks (red spheres)
                pos_obj = f"positive_peaks_{threshold_value}_{threshold_type}"
                for _, peak in pos_peaks.iterrows():
                    cmd.pseudoatom(pos_obj, pos=[peak['peakx'], peak['peaky'], peak['peakz']], 
                                 b=abs(peak['peak']), vdw=0.3)
                cmd.color("red", pos_obj)
                cmd.show("spheres", pos_obj)
                
                # Carve around all symmetry-equivalent models
                cmd.select(f"temp_sel_{pos_obj}", f"{pos_obj} within {distance_cutoff*1.5} of ({model_selection})")
                if cmd.count_atoms(f"temp_sel_{pos_obj}") > 0:
                    cmd.create(f"{pos_obj}_carved", f"temp_sel_{pos_obj}")
                    cmd.delete(pos_obj)
                    cmd.set_name(f"{pos_obj}_carved", pos_obj)
                cmd.delete(f"temp_sel_{pos_obj}")
            
            if not neg_peaks.empty:
                # Create negative peaks (blue spheres)
                neg_obj = f"negative_peaks_{threshold_value}_{threshold_type}"
                for _, peak in neg_peaks.iterrows():
                    cmd.pseudoatom(neg_obj, pos=[peak['peakx'], peak['peaky'], peak['peakz']], 
                                 b=abs(peak['peak']), vdw=0.3)
                cmd.color("blue", neg_obj)
                cmd.show("spheres", neg_obj)
                
                # Carve around all symmetry-equivalent models
                cmd.select(f"temp_sel_{neg_obj}", f"{neg_obj} within {distance_cutoff*1.5} of ({model_selection})")
                if cmd.count_atoms(f"temp_sel_{neg_obj}") > 0:
                    cmd.create(f"{neg_obj}_carved", f"temp_sel_{neg_obj}")
                    cmd.delete(neg_obj)
                    cmd.set_name(f"{neg_obj}_carved", neg_obj)
                cmd.delete(f"temp_sel_{neg_obj}")
            
        except Exception as e:
            print(f"Error creating carved peak objects: {e}")

    def create_displacement_vectors_with_cgo(self, peaks_df, threshold_value, threshold_type):
        """Create displacement vectors using CGO arrows for single model display"""
        try:
            # Import CGO constants (these would normally come from PyMOL)
            try:
                from pymol import cgo
            except ImportError:
                # Define CGO constants if PyMOL is not available
                class CGO:
                    BEGIN = 0
                    END = 1
                    VERTEX = 2
                    NORMAL = 3
                    COLOR = 4
                    LINEWIDTH = 5
                    LINES = 6
                    TRIANGLES = 7
                    CYLINDER = 22
                    CONE = 23
                cgo = CGO()
            
            # Calculate per-atom weighted vectors
            atom_vectors = self.calculate_per_atom_vectors(peaks_df)
            
            if not atom_vectors:
                return
            
            vector_obj = f"displacement_vectors_{threshold_value}_{threshold_type}"
            scale_factor = self.vector_scale_spin.value()
            
            # Create CGO arrow objects for each atom
            all_cgo_objects = []
            
            for atom_key, vector_data in atom_vectors.items():
                atom_pos = vector_data['atom_pos']
                avg_vector = vector_data['avg_vector']
                
                # Calculate end position
                end_pos = [
                    atom_pos[0] + avg_vector[0] * scale_factor,
                    atom_pos[1] + avg_vector[1] * scale_factor,
                    atom_pos[2] + avg_vector[2] * scale_factor
                ]
                
                # Only show significant vectors
                magnitude = np.linalg.norm(avg_vector)
                if magnitude < 0.1:
                    continue
                
                # Create CGO arrow
                arrow_cgo = self.create_cgo_arrow(atom_pos, end_pos, magnitude)
                all_cgo_objects.extend(arrow_cgo)
            
            # Load all CGO objects as a single object
            if all_cgo_objects:
                try:
                    cmd.load_cgo(all_cgo_objects, vector_obj)
                    print(f"Created CGO displacement vectors: {vector_obj}")
                except NameError:
                    # PyMOL not available, just print what would be created
                    print(f"Would create CGO object '{vector_obj}' with {len(all_cgo_objects)} elements")
            
        except Exception as e:
            print(f"Error creating CGO displacement vectors: {e}")
            # Fallback to the original pseudoatom method
            self.create_displacement_vectors(peaks_df, [], threshold_value, threshold_type)

    def create_cgo_arrow(self, start_pos, end_pos, magnitude):
        """Create a CGO arrow from start to end position"""
        try:
            # Import CGO constants
            try:
                from pymol import cgo
            except ImportError:
                # Define CGO constants if PyMOL is not available
                class CGO:
                    BEGIN = 0
                    END = 1
                    VERTEX = 2
                    NORMAL = 3
                    COLOR = 4
                    LINEWIDTH = 5
                    LINES = 6
                    TRIANGLES = 7
                    CYLINDER = 22
                    CONE = 23
                cgo = CGO()
            
            arrow_objects = []
            
            # Calculate arrow properties
            arrow_vector = np.array(end_pos) - np.array(start_pos)
            arrow_length = np.linalg.norm(arrow_vector)
            
            if arrow_length < 1e-6:
                return []
            
            # Normalize vector for direction
            arrow_direction = arrow_vector / arrow_length
            
            # Calculate shaft end (90% of total length)
            shaft_length = arrow_length * 0.9
            shaft_end = np.array(start_pos) + arrow_direction * shaft_length
            
            # Color based on magnitude and direction
            if np.sum(arrow_vector) > 0:
                color = [1.0, 0.65, 0.0]  # Orange
            else:
                color = [0.0, 1.0, 1.0]  # Cyan
            
            # Cylinder shaft
            shaft_radius = min(0.05, arrow_length * 0.02)
            arrow_objects.extend([
                cgo.CYLINDER,
                start_pos[0], start_pos[1], start_pos[2],
                shaft_end[0], shaft_end[1], shaft_end[2],
                shaft_radius,
                color[0], color[1], color[2],
                color[0], color[1], color[2]
            ])
            
            # Cone arrowhead
            head_radius = shaft_radius * 2.5
            arrow_objects.extend([
                cgo.CONE,
                shaft_end[0], shaft_end[1], shaft_end[2],
                end_pos[0], end_pos[1], end_pos[2],
                head_radius, 0.0,
                color[0], color[1], color[2],
                color[0], color[1], color[2],
                1.0, 1.0
            ])
            
            return arrow_objects
            
        except Exception as e:
            print(f"Error creating CGO arrow: {e}")
            return []

    def create_displacement_vectors(self, peaks_df, model_objects, threshold_value, threshold_type):
        """Create per-atom displacement vectors based on weighted peak directions"""
        try:
            # Calculate per-atom weighted vectors
            atom_vectors = self.calculate_per_atom_vectors(peaks_df)
            
            if not atom_vectors:
                return
            
            vector_obj = f"displacement_vectors_{threshold_value}_{threshold_type}"
            scale_factor = self.vector_scale_spin.value()
            
            # Create arrow objects for each atom
            for atom_key, vector_data in atom_vectors.items():
                atom_pos = vector_data['atom_pos']
                avg_vector = vector_data['avg_vector']
                
                # Calculate end position
                end_pos = [
                    atom_pos[0] + avg_vector[0] * scale_factor,
                    atom_pos[1] + avg_vector[1] * scale_factor,
                    atom_pos[2] + avg_vector[2] * scale_factor
                ]
                
                # Create arrow using CGO (Compiled Graphics Objects)
                arrow_name = f"arrow_{atom_key.replace(':', '_').replace(' ', '_')}"
                
                # Create arrow shaft
                cmd.pseudoatom(f"{arrow_name}_start", pos=atom_pos, vdw=0.05)
                cmd.pseudoatom(f"{arrow_name}_end", pos=end_pos, vdw=0.1)
                
                # Color based on vector magnitude and direction
                magnitude = np.linalg.norm(avg_vector)
                if magnitude > 0.1:  # Only show significant vectors
                    if np.sum(avg_vector) > 0:
                        cmd.color("orange", f"{arrow_name}_*")
                    else:
                        cmd.color("cyan", f"{arrow_name}_*")
                    
                    # Create distance object to show as line
                    cmd.distance(f"{arrow_name}_line", f"{arrow_name}_start", f"{arrow_name}_end")
                    cmd.hide("labels", f"{arrow_name}_line")
                    
                    # Group all arrow components
                    cmd.group(vector_obj, f"{arrow_name}_*")
            
            # Hide individual components, show only lines
            cmd.hide("everything", f"{vector_obj}*")
            cmd.show("lines", f"{vector_obj}*line*")
            cmd.set("line_width", 3, f"{vector_obj}*line*")
            
        except Exception as e:
            print(f"Error creating displacement vectors: {e}")
    
    def calculate_per_atom_vectors(self, peaks_df):
        """Calculate weighted average displacement vectors per atom using symmetry-corrected coordinates"""
        atom_vectors = {}
        
        # Validate required columns exist
        required_columns = ['coordx', 'coordy', 'coordz', 'chain', 'seqid', 'residue', 'atom', 'altloc', 
                           'peakx', 'peaky', 'peakz', 'markx', 'marky', 'markz', 'peak']
        missing_columns = [col for col in required_columns if col not in peaks_df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns for vector calculation: {missing_columns}")
            return {}
        
        if peaks_df.empty:
            print("Warning: Empty peaks DataFrame, no vectors to calculate")
            return {}
        
        # Group by the original atom coordinates (coordx, coordy, coordz)
        # but use mark coordinates for vector calculations
        try:
            grouped = peaks_df.groupby(['coordx', 'coordy', 'coordz', 'chain', 'seqid', 'residue', 'atom', 'altloc'])
        except KeyError as e:
            print(f"Error: Cannot group by required columns: {e}")
            return {}
        
        for (coord_x, coord_y, coord_z, chain, seqid, residue, atom, altloc), group in grouped:
            # Create unique atom identifier
            atom_key = f"{chain}:{seqid}:{residue}:{atom}"
            
            if altloc is not None and altloc != 'None':
                atom_key += f":{altloc}"
            
            # Calculate weighted average vector using mark positions
            weighted_vectors = []
            total_weight = 0
            
            for _, peak in group.iterrows():
                # Weight by peak height (including sign)
                weight = peak['peak']  # This includes the sign
                
                # Calculate vector from original atom position to peak position
                # This anchors the visualization to the model
                vector = np.array([
                    peak['peakx'] - coord_x,
                    peak['peaky'] - coord_y, 
                    peak['peakz'] - coord_z
                ])
                
                weighted_vectors.append(weight * vector)
                total_weight += abs(weight)
            
            if total_weight > 0:
                # Average the weighted vectors
                avg_vector = np.sum(weighted_vectors, axis=0) / len(weighted_vectors)
                
                # Use original atom coordinates as the display position (anchor to model)
                display_pos = [coord_x, coord_y, coord_z]
                
                atom_vectors[atom_key] = {
                    'atom_pos': display_pos,  # Use original atom position for display
                    'avg_vector': avg_vector,
                    'total_weight': total_weight,
                    'num_peaks': len(group)
                }
        
        return atom_vectors
    
    def display_analytics_summary(self, analytics):
        """Display a summary of vector analytics"""
        try:
            print("Displaying analytics summary...")
            summary = analytics['summary']
            
            summary_text = f"""Vector Analysis Summary:

Total atoms with vectors: {analytics['total_atoms']}
Mean displacement magnitude: {summary['mean_magnitude']:.3f} Å
Standard deviation: {summary['std_magnitude']:.3f} Å
Maximum displacement: {summary['max_magnitude']:.3f} Å
Minimum displacement: {summary['min_magnitude']:.3f} Å
Total displacement: {summary['total_displacement']:.3f} Å

Chain analysis:"""
            
            for chain, data in analytics['chain_analysis'].items():
                chain_mags = np.array(data['magnitudes'])
                summary_text += f"""
Chain {chain}: {len(data['residues'])} residues, mean displacement: {np.mean(chain_mags):.3f} Å"""
            
            # Show the analytics dialog
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("Vector Analytics")
            msg_box.setText(summary_text)
            msg_box.setIcon(QtWidgets.QMessageBox.Information)
            msg_box.exec_()
            
            print("Analytics summary displayed successfully")
            
        except Exception as e:
            print(f"Error displaying analytics summary: {e}")
            import traceback
            traceback.print_exc()
            
            analytics = {
                'total_atoms': len(atom_vectors),
                'vector_magnitudes': [],
                'residue_analysis': {},
                'chain_analysis': {}
            }
            
            for atom_key, vector_data in atom_vectors.items():
                # Parse atom key
                parts = atom_key.split(':')
                chain = parts[0]
                seqid = parts[1] 
                residue = parts[2]
                atom = parts[3]
                
                magnitude = np.linalg.norm(vector_data['avg_vector'])
                analytics['vector_magnitudes'].append(magnitude)
                
                # Per residue analysis
                res_key = f"{chain}:{seqid}:{residue}"
                if res_key not in analytics['residue_analysis']:
                    analytics['residue_analysis'][res_key] = {
                        'atoms': [],
                        'magnitudes': [],
                        'vectors': []
                    }
                
                analytics['residue_analysis'][res_key]['atoms'].append(atom)
                analytics['residue_analysis'][res_key]['magnitudes'].append(magnitude)
                analytics['residue_analysis'][res_key]['vectors'].append(vector_data['avg_vector'])
                
                # Per chain analysis
                if chain not in analytics['chain_analysis']:
                    analytics['chain_analysis'][chain] = {
                        'residues': set(),
                        'magnitudes': [],
                        'vectors': []
                    }
                
                analytics['chain_analysis'][chain]['residues'].add(res_key)
                analytics['chain_analysis'][chain]['magnitudes'].append(magnitude)
                analytics['chain_analysis'][chain]['vectors'].append(vector_data['avg_vector'])
            
            # Calculate summary statistics
            magnitudes = np.array(analytics['vector_magnitudes'])
            analytics['summary'] = {
                'mean_magnitude': np.mean(magnitudes),
                'std_magnitude': np.std(magnitudes),
                'max_magnitude': np.max(magnitudes),
                'min_magnitude': np.min(magnitudes),
                'total_displacement': np.sum(magnitudes)
            }
            
            return analytics
            
        except Exception as e:
            print(f"Error generating vector analytics: {e}")
            return None

# Import the original IADDAT functions with bug fixes
def map_threshold(realmap, threshold, cell, negative=False):
    """Find every grid point above threshold in map."""
    # Fixed: Use correct function name from skimage
    peaks = feature.peak_local_max(realmap, threshold_abs=threshold, exclude_border=False)
    data = []
    for p in peaks:
        pf = p/np.array(realmap.shape)
        pos = cell.orthogonalize(gemmi.Fractional(*pf))
        d = {"x": pos.x, "y": pos.y, "z": pos.z}
        if negative:
            d["height"] = realmap[p[0], p[1], p[2]]*-1
        else:
            d["height"] = realmap[p[0], p[1], p[2]]
        data.append(d)
    return pd.DataFrame(data)

def IADDAT_peaks_table(input_PDB_filename, input_MTZ_filename, input_column_labels="FoFo,PHFc", 
                      threshold_value=3.0, threshold_type="sigma", distance_cutoff=1.2):
    """Tabulate absolute difference density at a defined threshold within a defined cutoff distance of a model."""
    input_PDB = gemmi.read_structure(input_PDB_filename)
    input_PDB.remove_hydrogens()
    input_PDB.remove_empty_chains()
    cell = input_PDB.cell
    model = input_PDB[0]
    input_MTZ = gemmi.read_mtz_file(input_MTZ_filename)
    
    try:
        column_labels = [input_column_labels.split(',')[0], input_column_labels.split(',')[1]]
        mtz_fphi = input_MTZ.get_f_phi(column_labels[0], column_labels[1])
    except:
        raise ValueError(f"Error: please provide column labels corresponding to structure factors and phases; eg: 'FoFo,PHFc'\nColumns in file: {input_MTZ.column_labels()}")
    
    sf = mtz_fphi.transform_f_phi_to_map(sample_rate=4)
    # Use full unit cell map instead of ASU-masked map to find peaks throughout the cell
    realmap = np.array(sf, copy=False)
    
    if threshold_type == "sigma":
        sites_pos = map_threshold(realmap, realmap.std()*threshold_value, input_MTZ.cell, False)
        sites_neg = map_threshold(realmap*-1, realmap.std()*threshold_value, input_MTZ.cell, True)
    elif threshold_type == "absolute":
        sites_pos = map_threshold(realmap, threshold_value, input_MTZ.cell, False)
        sites_neg = map_threshold(realmap*-1, threshold_value, input_MTZ.cell, True)
    else:
        raise ValueError("Error: please select valid threshold_type: 'sigma' or 'absolute'")
    
    sites_all = pd.concat([sites_pos, sites_neg])
    ns = gemmi.NeighborSearch(model, cell, distance_cutoff).populate()
    peaks = []
    
    for idx, peak in sites_all.iterrows():
        blob = gemmi.Position(peak.x, peak.y, peak.z)
        marks = ns.find_atoms(blob)
        if len(marks) == 0:
            continue

        for mark in marks:
            cra = mark.to_cra(model)
            dist = cell.find_nearest_pbc_image(blob, cra.atom.pos, mark.image_idx).dist()
            COM = cra.chain.calculate_center_of_mass()
            
            temp_altloc = cra.atom.altloc if cra.atom.has_altloc() else None
            
            record = {
                "chain": cra.chain.name,
                "seqid": cra.residue.seqid.num,
                "residue": cra.residue.name,
                "atom": cra.atom.name,
                "altloc": temp_altloc,
                "dist": dist,
                "peak": peak.height,
                "coordx": cra.atom.pos.x,
                "coordy": cra.atom.pos.y,
                "coordz": cra.atom.pos.z,
                "peakx": peak.x,
                "peaky": peak.y,
                "peakz": peak.z,
                "markx": mark.pos.x,
                "marky": mark.pos.y,
                "markz": mark.pos.z,
                "image_idx": mark.image_idx,
                "mol_COM": COM.tolist(),
                "deltax": peak.x - mark.pos.x,
                "deltay": peak.y - mark.pos.y,
                "deltaz": peak.z - mark.pos.z,
            }
            peaks.append(record)
    
    # Create DataFrame and validate it has expected columns
    df = pd.DataFrame.from_records(peaks)
    
    # Debug output
    print(f"Debug: Created peaks DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    
    # If no peaks found, create empty DataFrame with required columns
    if df.empty:
        print("Warning: No peaks found within distance cutoff")
        required_columns = ["chain", "seqid", "residue", "atom", "altloc", "dist", "peak", 
                           "coordx", "coordy", "coordz", "peakx", "peaky", "peakz", 
                           "markx", "marky", "markz", "image_idx", "mol_COM", "deltax", "deltay", "deltaz"]
        df = pd.DataFrame(columns=required_columns)
    
    return df

def IADDAT_integrator(IADDAT_peaks_df, input_PDB_filename, input_MTZ_filename, 
                     threshold_value=3.0, threshold_type="sigma", distance_cutoff=1.2):
    """Integrate IADDAT values and save results."""
    
    # Validate DataFrame and add debugging
    print(f"Debug: IADDAT_peaks_df shape: {IADDAT_peaks_df.shape}")
    print(f"Debug: IADDAT_peaks_df columns: {list(IADDAT_peaks_df.columns)}")
    
    # Check for required columns
    required_columns = ['coordx', 'coordy', 'coordz', 'peak']
    missing_columns = [col for col in required_columns if col not in IADDAT_peaks_df.columns]
    
    if missing_columns:
        raise KeyError(f"Missing required columns in DataFrame: {missing_columns}. Available columns: {list(IADDAT_peaks_df.columns)}")
    
    if IADDAT_peaks_df.empty:
        print("Warning: IADDAT_peaks_df is empty. No peaks found.")
        # Create empty output files with warning
        pdb_string = os.path.basename(input_PDB_filename).replace('.pdb', '')
        mtz_string = os.path.basename(input_MTZ_filename).replace('.mtz', '')
        output_pdb_string = f"{pdb_string}_{mtz_string}_total-IADDAT-in-B-iso-{threshold_value}-{threshold_type}_within-{distance_cutoff}-angstroms.pdb"
        df_excel_string = f"{pdb_string}_{mtz_string}_IADDAT-table-{threshold_value}-{threshold_type}_within-{distance_cutoff}-angstroms.xlsx"
        
        # Write original PDB without modifications
        input_PDB = gemmi.read_structure(input_PDB_filename)
        input_PDB.write_minimal_pdb(output_pdb_string)
        
        # Create empty IADDAT DataFrame
        empty_df = pd.DataFrame(columns=["chain", "residue_number", "residue_name", "atom_name", "atom_altloc", "IADDAT"])
        empty_df.to_excel(df_excel_string)
        return
    
    pdb_string = os.path.basename(input_PDB_filename).replace('.pdb', '')
    mtz_string = os.path.basename(input_MTZ_filename).replace('.mtz', '')
    output_pdb_string = f"{pdb_string}_{mtz_string}_total-IADDAT-in-B-iso-{threshold_value}-{threshold_type}_within-{distance_cutoff}-angstroms.pdb"
    df_excel_string = f"{pdb_string}_{mtz_string}_IADDAT-table-{threshold_value}-{threshold_type}_within-{distance_cutoff}-angstroms.xlsx"
    
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
                    # Fixed: Use proper column names and handle floating point comparison
                    coord_x = atom.pos.x
                    coord_y = atom.pos.y
                    coord_z = atom.pos.z
                    
                    # Use a tolerance for floating point comparison
                    tolerance = 0.001
                    peaks = IADDAT_peaks_df[
                        (abs(IADDAT_peaks_df['coordx'] - coord_x) < tolerance) &
                        (abs(IADDAT_peaks_df['coordy'] - coord_y) < tolerance) &
                        (abs(IADDAT_peaks_df['coordz'] - coord_z) < tolerance)
                    ]
                    
                    unique = np.unique(peaks['peak'].values) if not peaks.empty else np.array([])
                    total_IADDAT = np.abs(unique).sum()
                    per_resi_IADDAT.append(total_IADDAT)
                    atom.b_iso = total_IADDAT
                    
                    altloc = atom.altloc if atom.has_altloc() else "None"
                    IADDAT.append([chain.name, str(residue.seqid), residue.name, atom.name, altloc, total_IADDAT])
                    
                except Exception as e:
                    print(f"Warning: Error processing atom {chain.name}:{residue.seqid}:{residue.name}:{atom.name} - {e}")
                    # Set default values for problematic atoms
                    total_IADDAT = 0.0
                    per_resi_IADDAT.append(total_IADDAT)
                    atom.b_iso = total_IADDAT
                    
                    altloc = atom.altloc if atom.has_altloc() else "None"
                    IADDAT.append([chain.name, str(residue.seqid), residue.name, atom.name, altloc, total_IADDAT])
            
            per_resi_avg_IADDAT = np.array(per_resi_IADDAT).mean() if per_resi_IADDAT else 0.0
            for atom in residue:
                atom.occ = per_resi_avg_IADDAT
    
    input_PDB.write_minimal_pdb(output_pdb_string)
    iaddat_df = pd.DataFrame(IADDAT, columns=["chain", "residue_number", "residue_name", "atom_name", "atom_altloc", "IADDAT"])
    iaddat_df.to_excel(df_excel_string)

def run_plugin():
    """Launch the IADDAT plugin"""
    dialog = IADDATPlugin()
    dialog.show()
    dialog.exec_()