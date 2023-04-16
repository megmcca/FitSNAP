from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.parallel_tools import ParallelTools


pt = ParallelTools()


class CheckTraining(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowed_keys = ['mode', 'vars_mode']
        self.allowed_modes = ['threshold', 'reference']
        self.vars_per_mode = []
        self.types = None

        if not config.has_section("CHECK_TRAINING"):
            self.delete()
            return

        # for value_name in config['REFERENCE']:
        #     if value_name in allowedkeys: continue
        #     else: pt.single_print(">>> Found unmatched variable in REFERENCE section of input: ",value_name)

        # Ensure mode input is valid
        self.modes = self.get_value("CHECK_TRAINING", "mode", "threshold").split()
        for mode in self.modes:
            if mode not in self.allowed_modes:
                pt.single_print(f">>> Found error in CHECK_TRAINING section of input: mode '{mode}' not recognized/implemented (current valid modes: threshold, reference)")
                return

        # Check values match atom types for 'threshold' and 'reference' modes
        if 'threshold' in self.modes or 'reference' in self.modes:
            if config.has_section("BISPECTRUM"):
                self.types = self.get_value("BISPECTRUM", "type", "H").split()
            elif config.has_section("ACE"):
                self.types = self.get_value("ACE", "type", "H").split()
            elif config.has_section("CUSTOM"):
                self.types = self.get_value("CUSTOM", "type", "H").split()

        # Get set of variables for each mode
        for name, value in self._config.items("CHECK_TRAINING"):
            if 'vars_mode' in name:
                self.vars_per_mode.append(value.split())
        
        # Check that each mode has input for variables
        if len(self.modes) != len(self.vars_per_mode):
            pt.single_print(f">>> Found error in CHECK_TRAINING section of input: number of modes does not match number of vars_mode inputs (expected {len(self.modes)}, found {len(self.vars_per_mode)})")
            return

        # Check mode variables for expected values (currently only # atom types)
        for i, mode in enumerate(self.modes):
            vars_mode = self.vars_per_mode[i]
            vars_mode_name = f'vars_mode{i+1}'
            if len(vars_mode) != len(self.types):
                pt.single_print(f">>> Found error in CHECK_TRAINING section of input: mode '{mode}' has mismatched variables (expected {len(self.types)}, found {len(vars_mode)} in {vars_mode_name})")
                return
        
        # TODO implement graceful handling of errors consistent with other Sections 
        # TODO create documentation for parent Section class and all child Sections (including this one)
        self.delete()
