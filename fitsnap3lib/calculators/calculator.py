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
    def check_training_data(self):
        @self.pt.rank_zero
        def decorated_check_training_data():
            # Currently only run if the '-nofit_trainingcheck' flag is used in the command line.
            # Executed in fitsnap.py.
            # 'Version 0' (this one) outputs two CSV files: one with statistical analysis performed on forces and energies 

            print("Command line flag '-nofit_trainingcheck' used, checking training data statistics (in beta!)")
            print("Current analysis type:  outside of 25% and 75% quartiles")
            print("Processing data...")

            pt = ParallelTools()

            # Grab relevant data from shared arrays/dicts and create base dataframe
            file_id_keys = 'Groups Configs Row_Type'.split()
            df = pd.DataFrame.from_dict({key:pt.fitsnap_dict[key] for key in file_id_keys})
            df['truths'] = pt.shared_arrays['b'].array.tolist()

            # Set up output
            all_output = []
            descr_cols = 'count mean std min 25th 50th 75th max'
            all_header = f'Groups Configs Row_Type {descr_cols} is_funky_all is_funky_group'.split()
            make_zero_if_tiny = lambda x: 0.0 if abs(x) < 10e-7 else x
            
            # Get user checking/fitting choices from calculator section of input file  
            # Use only those choices when compiling data
            chosen_row_types_input = [bool(val) for val in [self.config.sections['CALCULATOR'].energy, self.config.sections['CALCULATOR'].force, self.config.sections['CALCULATOR'].stress]]
            chosen_row_types = [row_type for i, row_type in enumerate('Energy Force Stress'.split()) if chosen_row_types_input[i]]

            for row_type in chosen_row_types:
                # Collect data for entire dataset for row type
                df_row_type = df.loc[df.Row_Type == row_type,:]
                vals_row_type = df_row_type.truths.describe().values
                mean_all = make_zero_if_tiny(vals_row_type[1])
                std_all = vals_row_type[2]
                lo_q_all = vals_row_type[4]
                hi_q_all = vals_row_type[6]
                is_funky_all = self._check_funkiness_quartiles(mean_all, lo_q_all, hi_q_all)
                str_row = ' '.join([str(v) for v in vals_row_type])
                data_row_all =  f'all all {row_type} {str_row}'.split() + [is_funky_all, False]
                all_output.append(data_row_all)

                # Per group/atom_type, also collect per-group quartile data to compare configs to
                # group_quartiles = [] # df version
                group_quartiles = {}
                for label, subset in df_row_type.groupby('Groups'):
                    vals_group = subset.truths.describe().values.tolist()
                    mean_group, lo_q_group, hi_q_group = make_zero_if_tiny(vals_group[1]), vals_group[4], vals_group[6]
                    # group_quartiles.append((name[0], name[1], lo_q_group, hi_q_group))
                    group_quartiles[label] = (lo_q_group, hi_q_group)
                    is_funky_all = self._check_funkiness_quartiles(mean_group, lo_q_all, hi_q_all)
                    row_group = [label, 'all', row_type] + vals_group + [is_funky_all, False]
                    all_output.append(row_group)

                # Per config with comparison to group quartiles
                for label, subset in df_row_type.groupby('Groups Configs'.split()):
                    vals_config = subset.truths.describe().values.tolist()
                    mean_config = make_zero_if_tiny(vals_config[1])
                    is_funky_all = self._check_funkiness_quartiles(mean_config, lo_q_all, hi_q_all)
                    lo_q_group, hi_q_group = group_quartiles[label[0]]
                    is_funky_group = self._check_funkiness_quartiles(mean_config, lo_q_group, hi_q_group)
                    row_config = [label[0], label[1], row_type] + vals_config + [is_funky_all, is_funky_group]
                    all_output.append(row_config)
                    print(row_config)
                    exit()
                del df_row_type

            # Consolidate
            df_out = pd.DataFrame.from_records(all_output, columns=all_header)
            total_configs = df_out[(df_out.Row_Type == 'Energy')&(df_out.Configs != 'all')].shape[0]

            all_funky_ones = []
            all_funky_counts = []
            for row_type in chosen_row_types:
                mask_funky_config = ((df_out.Row_Type == row_type)&(df_out.Configs != 'all'))&((df_out.is_funky_all)|(df_out.is_funky_group))
                funky_ones = df_out.loc[mask_funky_config,:]
                all_funky_ones.append(funky_ones)
                all_funky_counts.append(funky_ones.shape[0])

            df_funky = pd.concat(all_funky_ones)
            df_funky_short = df_funky.loc[:,'Groups Configs Row_Type is_funky_all is_funky_group'.split()]

            # Output some simple statistics
            print("... analysis complete!")
            print(f"Report on funky configs per fitting type (energy, force, stress),  out of {total_configs} total configs:")
            
            for i, row_type in enumerate(chosen_row_types):
                row_str = f'\t{row_type}: {all_funky_counts[i]}'
                print(row_str)
            
            print("Groups with funky configs: ")
            for funky_group in df_funky.Groups.unique():
                print(f'\t{funky_group}')

            print("Writing CSVs...")
            df_out_csv_all = 'NoFit_TrainingCheck_all.csv'
            df_out_csv_funky = 'NoFit_TrainingCheck_funky.csv'
            df_out_csv_funky_short = 'NoFit_TrainingCheck_funky-short.csv'
            df_out.to_csv(df_out_csv_all, index=False)
            df_funky.to_csv(df_out_csv_funky, index=False)
            df_funky_short.to_csv(df_out_csv_funky_short, index=False)
            print("Data written, training set check complete")
            del df
        decorated_check_training_data()

    def _check_funkiness_quartiles(self, mean, lo_q, hi_q, label=''):
        # If the mean is not within the 2nd or 3rd quartiles of the queried aggregated data, it's funky (in a bad way)
        if mean < lo_q or mean > hi_q:
            return True
        
        # Implement more sophisticated tests and return statements here
        return False
    
    # def _check_funkiness_thresholds(self, val, lo_thresh, hi_thresh, label=''):
    #     # TODO implement
    #     if val < lo_thresh or val > hi_thresh:
    #         return True

    #     # Implement more sophisticated tests and return statements here
    #     return False

    # def _check_funkiness_Edeltas(self, val, lo_thresh, hi_thresh, label=''):
    #     # TODO implement - get expected per-atom energies (cohesive) from user (INPUT FILE) to shift distributions so that outliers look properly funky
    #     if val < lo_thresh or val > hi_thresh:
    #         return True

    #     # Implement more sophisticated tests and return statements here
    #     return False
