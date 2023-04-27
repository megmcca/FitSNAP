import numpy as np
import pandas as pd
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
import matplotlib.pyplot as plt

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