from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.parallel_tools import ParallelTools


pt = ParallelTools()


class CheckTraining(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowed_keys = ['mode', 'vars_mode']
        self.allowed_modes = ['threshold', 'reference']
        self.modes_columns = [["thr_E","thr_above_E","thr_F","thr_outside_F","thr_sigma","thr_outside_sigma"], ["ref_config_E","ref_calc_E","ref_dE","ref_thresh_dE","ref_above_dE"]]
        self.modes_flag_columns = [["thr_above_E","thr_outside_F","thr_outside_sigma"], ["ref_above_dE"]]
        self.modes_columns_units = [["eV", "-","eV/A","-","GPa","-"], ["eV","eV","eV","eV","-"]]
        self.vars_per_mode = []
        self.vars_per_mode_units = []
        self.vars_per_mode_labels = []
        self.vars_per_mode_columns = []
        self.chosen_row_types_input = []
        self.atom_types = []
        self.has_valid_input = False

        if not config.has_section("CHECKTRAINING"):
            self.delete()
            return

        # for value_name in config['REFERENCE']:
        #     if value_name in allowedkeys: continue
        #     else: pt.single_print(">>> Found unmatched variable in REFERENCE section of input: ",value_name)

        # Ensure mode input is valid
        self.modes = self.get_value("CHECKTRAINING", "mode", "threshold").split()
        for mode in self.modes:
            if mode not in self.allowed_modes:
                pt.single_print(f">>> Found error in CHECKTRAINING section of input: mode '{mode}' not recognized/implemented (current valid modes: threshold, reference)")
                return
            
        #     # Only keep units and columns if user is fitting training for that data
        #     self.modes_columns[0] = [self.modes_columns[0][i] for i, row_type_bool in enumerate(self.chosen_row_types_input) if row_type_bool]

        # Prepare information for 'reference' mode
        if 'reference' in self.modes:
            if config.has_section("BISPECTRUM"):
                self.atom_types = self.get_value("BISPECTRUM", "type", "H").split()
            elif config.has_section("ACE"):
                self.atom_types = self.get_value("ACE", "type", "H").split()
            elif config.has_section("CUSTOM"):
                self.atom_types = self.get_value("CUSTOM", "type", "H").split()

        # Get set of variables for each mode
        for name, value in self._config.items("CHECKTRAINING"):
            if 'vars_mode' in name:
                self.vars_per_mode.append(value.split())
        
        # Check that each mode has input for variables
        if len(self.modes) != len(self.vars_per_mode):
            pt.single_print(f">>> Found error in CHECKTRAINING section of input: number of modes does not match number of vars_mode inputs (expected {len(self.modes)}, found {len(self.vars_per_mode)})")
            return
    
        # Remove info for fitting types ('row_types' in FitSNAP.df) if toggled off in input file
        etuple, ftuple, stuple = self.get_section("CALCULATOR")[1:]
        self.chosen_row_types_input = [bool(int(val)) for val in [etuple[1], ftuple[1],stuple[1]]]

        # Check mode variables for expected values and cast to proper datatype
        for i, mode in enumerate(self.modes):
            # Get variable name from input file for error output
            vars_mode_label = f'vars_mode{i+1}'
            self.vars_per_mode_labels.append(vars_mode_label)

            # Get variables
            vars_mode = self.vars_per_mode[i]
            self.vars_per_mode[i] = [float(val) if (val.lower() != 'none' and val.lower() != 'null') else None for val in vars_mode]
            
            # Threshold mode requires float or integer input for E, F, and sigma, but can take None or null 
            if mode == "threshold":
                if len(vars_mode) != 3:
                    pt.single_print(f">>> Found error in CHECKTRAINING section of input: mode 'threshold' has mismatched variables (expected 3 (E, F, and stress threshold, either can be 'None'), found {len(vars_mode)} in {vars_mode_label})")
                    return

            # Reference mode requires per atom type reference values (int or float) and a threshold for energy difference dE
            if mode == "reference":
                expected_nvars = len(self.atom_types) + 1
                if (len(vars_mode) != expected_nvars) or (None in self.vars_per_mode[i]):
                    pt.single_print(f">>> Found error in CHECKTRAINING section of input: mode 'reference' requires {expected_nvars} to match {self.atom_types} + one value for a threshold energy difference dE, found {len(vars_mode)} in {vars_mode_label})")
                    return
                
            # Assign other params, given user ordering
            self.vars_per_mode_columns.append(self.modes_columns[self.allowed_modes.index(mode)])
        
        # If we've gotten to this line, all input is valid
        # Used to feed back to fitsnap.py, to warn user if calculatur.check_training_data() would crash if run (rather tahn destroying an entire fit)
        self.has_valid_input = True
        
        # TODO implement graceful handling of errors consistent with other Sections 
        # TODO create documentation for parent Section class and all child Sections (including this one)
        self.delete()
