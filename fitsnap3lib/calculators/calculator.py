from fitsnap3lib.parallel_tools import ParallelTools, double_size, DistributedList, stubs
from fitsnap3lib.io.input import Config
from fitsnap3lib.io.output import output
import numpy as np
import pandas as pd


#config = Config()
#pt = ParallelTools()


class Calculator:

    def __init__(self, name):
        self.pt = ParallelTools()
        self.config = Config()
        self.name = name
        self.number_of_atoms = None
        self.number_of_files_per_node = None
        self.shared_index = None
        self.distributed_index = 0

    def get_width(self):
        pass

    def create_a(self):

        # TODO : Any extra config pulls should be done before this

        self.pt.sub_barrier()
        # total number of atoms in all configs, summed
        self.number_of_atoms = self.pt.shared_arrays["number_of_atoms"].array.sum()
        # total number of configs on all procs in a node
        self.number_of_files_per_node = len(self.pt.shared_arrays["number_of_atoms"].array)
        # self.nconfigs is the number of configs on this proc, assigned in lammps_base

        # create data matrices for nonlinear pytorch solver

        if (self.config.sections["SOLVER"].solver == "PYTORCH"):
            a_len = 0
            b_len = 0 # number reference energies for all configs
            c_len = 0 # number of reference forces for all configs
            dgrad_len = 0
            if self.config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if self.config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms # total number of atoms in all configs
                a_len += energy_rows
                b_len += self.number_of_files_per_node # total number of configs

            if self.config.sections["CALCULATOR"].force:
                c_len += 3*self.number_of_atoms
                dgrad_len += self.pt.shared_arrays["number_of_dgrad_rows"].array.sum()

            if self.config.sections["CALCULATOR"].per_atom_scalar:

                # in this case we fitting NNs only to per-atom scalars, not to energies/forces

                a_len += self.number_of_atoms # total number of atoms in all configs


#            # stress fitting not supported yet.
#            if config.sections["CALCULATOR"].stress:
#                a_len += self.number_of_files_per_node * 6
#                b_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)

            a_size = ( (a_len * a_width) + (dgrad_len * a_width) ) * double_size
            output.screen(">>> Matrix of descriptors and descriptor derivatives takes up ", 
                          "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                          "% of the total memory:", 
                          "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / self.pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / self.pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            self.pt.create_shared_array('a', a_len, a_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('b', b_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('c', c_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('w', b_len, 2, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('t', a_len, 1, tm=self.config.sections["SOLVER"].true_multinode)
            if self.config.sections["CALCULATOR"].per_atom_scalar:
                # create per-atom scalar arrays
                self.pt.create_shared_array('pas', a_len, 1, tm=self.config.sections["SOLVER"].true_multinode)

            #if self.config.sections["CALCULATOR"].force:
            self.pt.create_shared_array('dgrad', dgrad_len, a_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('dbdrindx', dgrad_len, 3, 
                                        tm=self.config.sections["SOLVER"].true_multinode)

            # make an index for which the 'a' array starts on a particular proc
            self.pt.new_slice_a()
            self.shared_index = self.pt.fitsnap_dict["sub_a_indices"][0] 
            # make an index for which the 'b' array starts on a particular proc
            self.pt.new_slice_b()
            self.shared_index_b = self.pt.fitsnap_dict["sub_b_indices"][0] 
            # make an index for which the 'c' array starts on a particular proc
            self.pt.new_slice_c()
            self.shared_index_c = self.pt.fitsnap_dict["sub_c_indices"][0] 
            #self.pt.new_slice_t() # atom types

            # make an index for which the 'dgrad' array starts on a particular proc
            self.pt.new_slice_dgrad()
            self.shared_index_dgrad = self.pt.fitsnap_dict["sub_dgrad_indices"][0]

            # create fitsnap dicts - distributed lists of size nconfig per proc
            # these later get gathered on the root proc in calculator.gather_distributed_lists

            self.pt.add_2_fitsnap("Groups", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("Configs", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("NumAtoms", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("NumDgradRows", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("Testing", DistributedList(self.nconfigs))

        # get data arrays for network solvers

        elif (self.config.sections["SOLVER"].solver == "NETWORK"):

            a_len = 0 # per-atom quantities (types, numneighs) for all configs
            b_len = 0 # number reference energies for all configs
            c_len = 0 # number of reference forces for all configs
            c_width = 0 # 3 if fitting to forces
            neighlist_len = 0 # number of neighbors for all configs

            a_len += self.number_of_atoms # total number of atoms in all configs
            neighlist_len += self.pt.shared_arrays["number_of_neighs_scrape"].array.sum()

            if self.config.sections["CALCULATOR"].energy:
                b_len += self.number_of_files_per_node # total number of configs

            if self.config.sections["CALCULATOR"].force:
                c_len += 3*self.number_of_atoms

#            # stress fitting not supported yet.
#            if config.sections["CALCULATOR"].stress:
#                a_len += self.number_of_files_per_node * 6
#                b_len += self.number_of_files_per_node * 6

            a_width = 2 # types and numneighs
            neighlist_width = self.get_width()
            assert isinstance(neighlist_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
            a_size = (neighlist_len * neighlist_width + 2 * c_len * c_width) * double_size
            output.screen(">>> Matrix of data takes up ", "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                          "% of the total memory:", "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / self.pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The data memory larger than 50% of your RAM. \n Aborting...!")
            elif a_size / self.pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            # create shared arrays
            a_width = 5
            neighlist_width = 2 # i j 
            xneigh_width = 3 # xj yj zj, with PBC corrections
            self.pt.create_shared_array('a', a_len, a_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('neighlist', neighlist_len, neighlist_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('xneigh', neighlist_len, xneigh_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('transform_x', neighlist_len, xneigh_width, 
                                        tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('b', b_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('x', c_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('w', b_len, 2, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('t', a_len, 1, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('positions', a_len, 3, tm=self.config.sections["SOLVER"].true_multinode)
            
            # also need descriptors for network standardization
            # for pairwise networks, there are num_neigh*num_descriptors total descriptors to store
            # TODO: if statement here to catch possibilities for custom networks, e.g. nonpairwise descriptors, etc.

            self.pt.create_shared_array('descriptors', neighlist_len, self.config.sections['CUSTOM'].num_descriptors)

            if self.config.sections["CALCULATOR"].force:
                self.pt.create_shared_array('c', c_len, tm=self.config.sections["SOLVER"].true_multinode)

            # make an index for which the 'a' array starts on a particular proc
            self.pt.new_slice_a()
            self.shared_index = self.pt.fitsnap_dict["sub_a_indices"][0] 
            # make an index for which the 'b' array starts on a particular proc
            self.pt.new_slice_b()
            self.shared_index_b = self.pt.fitsnap_dict["sub_b_indices"][0] 
            # make an index for which the 'c' array starts on a particular proc
            self.pt.new_slice_c()
            self.shared_index_c = self.pt.fitsnap_dict["sub_c_indices"][0] 
            #self.pt.new_slice_t() # atom types
            # make an index for which the 'neighlist' array starts on a particular proc
            self.pt.new_slice_neighlist()
            self.shared_index_neighlist = self.pt.fitsnap_dict["sub_neighlist_indices"][0] 

            # create fitsnap dicts - distributed lists of size nconfig per proc
            # these later get gathered on the root proc in calculator.gather_distributed_lists

            self.pt.add_2_fitsnap("Groups", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("Configs", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("NumAtoms", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("NumNeighs", DistributedList(self.nconfigs))
            self.pt.add_2_fitsnap("Testing", DistributedList(self.nconfigs))

        # get data arrays for linear solvers

        else:

            a_len = 0
            if self.config.sections["CALCULATOR"].energy:
                energy_rows = self.number_of_files_per_node
                if self.config.sections["CALCULATOR"].per_atom_energy:
                    energy_rows = self.number_of_atoms
                a_len += energy_rows
            if self.config.sections["CALCULATOR"].force:
                a_len += 3 * self.number_of_atoms
            if self.config.sections["CALCULATOR"].stress:
                a_len += self.number_of_files_per_node * 6

            a_width = self.get_width()
            assert isinstance(a_width, int)

            # TODO: Pick a method to get RAM accurately (pt.get_ram() seems to get RAM wrong on Blake)
            a_size = a_len * a_width * double_size
            output.screen(">>> Matrix of descriptors takes up ", "{:.4f}".format(100 * a_size / self.config.sections["MEMORY"].memory),
                          "% of the total memory:", "{:.4f}".format(self.config.sections["MEMORY"].memory*1e-9), "GB")
            if a_size / self.pt.get_ram() > 0.5 and not self.config.sections["MEMORY"].override:
                raise MemoryError("The descriptor matrix is larger than 50% of your RAM. \n Aborting...!")
            elif a_size / self.pt.get_ram() > 0.5 and self.config.sections["MEMORY"].override:
                output.screen("Warning: I hope you know what you are doing!")

            self.pt.create_shared_array('a', a_len, a_width, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('b', a_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.create_shared_array('w', a_len, tm=self.config.sections["SOLVER"].true_multinode)
            #self.pt.create_shared_array('ref', a_len, tm=self.config.sections["SOLVER"].true_multinode)
            self.pt.new_slice_a()
            self.shared_index = self.pt.fitsnap_dict["sub_a_indices"][0]
            # pt.slice_array('a')

            self.pt.add_2_fitsnap("Groups", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Configs", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Row_Type", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Atom_I", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Testing", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))
            self.pt.add_2_fitsnap("Atom_Type", DistributedList(self.pt.fitsnap_dict["sub_a_size"]))

    def process_configs(self, data, i):
        pass

    def preprocess_configs(self, data, i):
        pass

    def preprocess_allocate(self, nconfigs):
        pass

    @staticmethod
    def collect_distributed_lists():
        """
        Gathers all the distributed lists on each proc to the root proc.
        For each distributed list (fitsnap dicts) this will create a concatenated list on the root proc.
        We use this function in fitsnap.py after processing configs.
        """
        pt = ParallelTools()    
        for key in pt.fitsnap_dict.keys():
            if isinstance(pt.fitsnap_dict[key], DistributedList):
                pt.gather_fitsnap(key)
                if pt.fitsnap_dict[key] is not None and stubs != 1:
                    pt.fitsnap_dict[key] = [item for sublist in pt.fitsnap_dict[key] for item in sublist]
                elif pt.fitsnap_dict[key] is not None:
                    pt.fitsnap_dict[key] = pt.fitsnap_dict[key].get_list()

    #@pt.rank_zero
    def extras(self):
        @self.pt.rank_zero
        def decorated_extras():
            pt = ParallelTools()
            config = Config()
            if config.sections["EXTRAS"].dump_a:
                np.save(config.sections['EXTRAS'].descriptor_file, pt.shared_arrays['a'].array)
            if config.sections["EXTRAS"].dump_b:
                np.save(config.sections['EXTRAS'].truth_file, pt.shared_arrays['b'].array)
            if config.sections["EXTRAS"].dump_w:
                np.save(config.sections['EXTRAS'].weights_file, pt.shared_arrays['w'].array)
            if config.sections["EXTRAS"].dump_dataframe:
                df = pd.DataFrame(pt.shared_arrays['a'].array)
                df['truths'] = pt.shared_arrays['b'].array.tolist()
                df['weights'] = pt.shared_arrays['w'].array.tolist()
                for key in pt.fitsnap_dict.keys():
                    if isinstance(pt.fitsnap_dict[key], list) and len(pt.fitsnap_dict[key]) == len(df.index):
                        df[key] = pt.fitsnap_dict[key]
                df.to_pickle(config.sections['EXTRAS'].dataframe_file)
                del df
        decorated_extras()

        # if not config.sections["SOLVER"].detailed_errors:
        #     print(
        #         ">>>Enable [SOLVER], detailed_errors = 1 to characterize the training/testing split of your output *.npy matricies")

    #@pt.rank_zero
    def check_training_data(self, make_pretty_CSVs=True):
        @self.pt.rank_zero
        def decorated_check_training_data():
            # Only executes if [CHECKTRAINING] section present in FitSNAP input file.
            # Executed in fitsnap.py.
            # Outputs three CSV files in the CHECKTRAINING section of the input file: one with all statistical analysis, and two containing only files flagged as outliers in a long (all columns) and 'human-readable' (only group/filename columns) formats
            # Analysis is performed on three levels: first on all (global) data, then subsets of data per group, then per config
            # Data that exists outside of the mode of analysis (right now, 'threshold' and 'reference') is flagged as "funky"
            # Per config data is checked against both global data (variable "is_funky_all") and within its own group (variable "is_funky_group")

            # To access shared array
            pt = ParallelTools()

            # Grab modes and mode variables from CHECKTRAINING section of input
            modes = self.config.sections["CHECKTRAINING"].modes
            vars_per_mode = self.config.sections["CHECKTRAINING"].vars_per_mode
            vars_per_mode_units = self.config.sections["CHECKTRAINING"].vars_per_mode_units
            vars_per_mode_labels = self.config.sections["CHECKTRAINING"].vars_per_mode_labels
            vars_per_mode_columns = self.config.sections["CHECKTRAINING"].vars_per_mode_columns

            # Grab relevant data from shared arrays/dicts and create base dataframe
            file_id_keys = 'Groups Configs Row_Type Atom_Type'.split()
            df = pd.DataFrame.from_dict({key:pt.fitsnap_dict[key] for key in file_id_keys})
            df['truths'] = pt.shared_arrays['b'].array.tolist()

            # Get user checking/fitting choices from calculator section of input file  
            # Use only those choices when compiling data
            chosen_row_types_input = [bool(val) for val in [self.config.sections['CALCULATOR'].energy, self.config.sections['CALCULATOR'].force, self.config.sections['CALCULATOR'].stress]]
            chosen_row_types = [row_type for i, row_type in enumerate('Energy Force Stress'.split()) if chosen_row_types_input[i]]

            # Set up output
            all_output = []
            all_mode_cols = []
            for i, mode in enumerate(modes):
                mode_cols = vars_per_mode_columns[i]
                all_mode_cols.append(mode_cols)

            all_config_dfs = []
            for i, mode in enumerate(modes):
                vars_mode = vars_per_mode[i]
                mode_cols = all_mode_cols[i]
                if mode == 'threshold':
                    all_config_dfs.append(self._get_mode_threshold_df(df, chosen_row_types, vars_mode, mode_cols))
                if mode == 'reference':
                    all_config_dfs.append(self._get_mode_reference_df(df, vars_mode, mode_cols))
                
            # TODO add option for user to add funky config flag to FitSNAP.df, for now creating new CSV/DataFrame

            # Merge dataframe on indices
            df_out = pd.concat(all_config_dfs,axis=1)
            df_out.index.name = 'config'

            # Start conditioning/cleaning output
            # Flatten to allow search
            flat_flag_cols = [item for subset in self.config.sections["CHECKTRAINING"].modes_flag_columns for item in subset]
            flag_cols = [col for col in flat_flag_cols if col in df_out.columns]
            if make_pretty_CSVs:
                make_pretty_round_precision = 5
                make_pretty_round_cols = [col for col in df_out.columns if col not in flag_cols]
                df_out.loc[:, make_pretty_round_cols] = df_out.loc[:,make_pretty_round_cols].round(make_pretty_round_precision)
                df_out.loc[:, flag_cols] = df_out.loc[:,flag_cols].astype(bool)
                
            
            # Flag outlier look at class vars in io/sections/checktraining.py for columns
            df_bool = df_out.loc[:,flag_cols] != 0 # .any(axis=1)
            df_short = df_out.loc[df_bool.any(axis=1),flag_cols]
            df_funky = df_out.loc[df_short.index,:]

            # Brief report
            nconfigs = df_out.shape[0]
            nfunky = df_short.shape[0]
            pt.single_print("Data processing complete.")
            pt.single_print(f"Of {nconfigs} configs analyzed, {nfunky} flagged as 'funky'")
            pt.single_print("Flagged configuration counts per variable: ")
            pt.single_print(df_bool.sum())
            
            # Write CSVs
            pt.single_print("Writing CSVs...")
            mode_str = "-".join(modes)
            df_out_csv_all = f'CheckTraining_{mode_str}_all-configs.csv'
            df_out_csv_funky = f'CheckTraining_{mode_str}_funky-configs.csv'
            df_out_csv_funky_short = f'CheckTraining_{mode_str}_short-funky-configs.csv'
            df_out.to_csv(df_out_csv_all, index=True)
            df_funky.to_csv(df_out_csv_funky, index=True)
            df_short.to_csv(df_out_csv_funky_short, index=True)
            pt.single_print("Data written, training set check complete")
            del df
        decorated_check_training_data()

    def _get_mode_threshold_df(self, df, row_types, thresh_vals, thresh_cols):
        # Input: df, energy/force/stress rows, values of thresholds per row
        # Output: dataframe from dictionary with threshold data per configuration
        # If the row type is energy, flag values outside of threshold as-is
        # If it's force or stress, take absolute values and flag those outside of threshold 
        # Current column in io/sections/checkraining.py: ["thresh_E","is_above_E","thresh_F","outside_F","thresh_sigma","outside_sigma"]
        mode_dict = {}
        gb_cols = 'Groups Configs'.split()
        final_cols = []
        
        # Get final column names for types of data checked
        for i, thresh_val in enumerate(thresh_vals):
            if thresh_val != None:
                j, k = i*2, i*2+2
                final_cols.extend(thresh_cols[j:k])
        
        # Dict format below is for new aggregated CheckTraining dataframe
        for label, subset in df.groupby(gb_cols):
            group_config = self._format_subset_label(label)
            mode_data = []
            for i, row_type in enumerate(row_types):
                thresh_val = thresh_vals[i]
                if thresh_val == None: continue            
                if row_type == 'Energy':
                    E = subset[subset.Row_Type == 'Energy'].truths.values[0]
                    is_above = 1 if E > thresh_val else 0
                    mode_data.extend([thresh_val, is_above])
                else:
                    list_bool = (subset[subset.Row_Type == row_type].truths.abs() > thresh_val).tolist()
                    nvals, ntrue = len(list_bool), np.sum(list_bool)
                    frac =  round(ntrue/nvals,5)
                    is_true = 1 if ntrue > 0 else 0
                    # mode_data.extend([thresh_val, frac]) # return fraction of force components
                    mode_data.extend([thresh_val, is_true]) # return simply "true"
            mode_dict[group_config] = mode_data

        mode_df = pd.DataFrame.from_dict(mode_dict, orient='index',columns=final_cols)
            
        # # Dict format below could be appended to FitSNAP.df as-is
        # for row_type in row_types:
        #     if thresh_val == None:
        #         continue
        #     elif row_type == 'Energy':
        #         mode_dict[thresh_col] = (df[df.Row_Type == 'Energy'].truths > thresh_val)
        #     else:
        #         mode_dict[thresh_col] = (df[df.Row_Type == row_type].truths.abs() > thresh_val)
        # mode_df = pd.DataFrame(mode_dict)

        return mode_df

    def _get_mode_reference_df(self, df, ref_vals, ref_cols):
        # Input: df, reference values per atom and a reference threshold for dE
        # Output: dataframe from dictionary with threshold data per configuration
        # Current column format in io/sections/checktraining.py: ["config_E","ref_E","dE","thresh_dE","is_above_dE"]
        mode_dict = {}
        per_atom_type_ref_E = ref_vals[:-1]
        thresh_dE = ref_vals[-1]

        for label, subset in df.groupby('Groups Configs'.split()):
            group_config = self._format_subset_label(label)
            config_E = subset[subset.Row_Type=='Energy'].truths.values[0]
            atom_type_counts = (subset[subset.Row_Type=='Force'].Atom_Type.value_counts()/3).astype(int).values
            atom_ref_energies = atom_type_counts*np.array(per_atom_type_ref_E)
            ref_E = np.sum(atom_ref_energies)/np.sum(atom_type_counts)
            dE = abs(config_E - ref_E)
            is_above_dE = 1 if dE > thresh_dE else 0
            mode_dict[group_config] = [config_E, ref_E, dE, thresh_dE, is_above_dE]

        mode_df = pd.DataFrame.from_dict(mode_dict, orient = 'index', columns = ref_cols)
        return mode_df
    
    def _format_subset_label(self, group_config_tuple):
        return "/".join(group_config_tuple)

