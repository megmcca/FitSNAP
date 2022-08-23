from fitsnap3lib.io.input import Config
from fitsnap3lib.parallel_tools import ParallelTools
import numpy as np
from pandas import DataFrame


#config = Config()
#pt = ParallelTools()


class Solver:

    def __init__(self, name, linear=True):
        self.config = Config()
        self.pt = ParallelTools()
        self.name = name
        self.configs = None
        self.fit = None
        self.all_fits = None
        self.template_error = False
        self.errors = []
        self.weighted = 'Unweighted'
        self.residuals = None
        self.a = None
        self.b = None
        self.w = None
        self.df = None
        self.linear = linear
        self._checks()

    def perform_fit(self):
        pass

    def fit_gather(self):
        # self.all_fits = pt.gather_to_head_node(self.fit)
        pass

    def _offset(self):
        num_types = self.config.sections["BISPECTRUM"].numtypes
        if num_types > 1:
            self.fit = self.fit.reshape(num_types, self.config.sections["BISPECTRUM"].ncoeff)
            offsets = np.zeros((num_types, 1))
            self.fit = np.concatenate([offsets, self.fit], axis=1)
            self.fit = self.fit.reshape((-1, 1))
        else:
            self.fit = np.insert(self.fit, 0, 0)

    def _checks(self):
        assert not (self.linear and self.config.sections['CALCULATOR'].per_atom_energy and self.config.args.perform_fit)

    #@pt.rank_zero
    def error_analysis(self):
        @self.pt.rank_zero
        def decorated_error_analysis():
            if not self.linear:
                self.pt.single_print("No Error Analysis for non-linear potentials")
                return
            self.df = DataFrame(self.pt.shared_arrays['a'].array)
            self.df['truths'] = self.pt.shared_arrays['b'].array.tolist()
            self.df['preds'] = self.pt.shared_arrays['a'].array @ self.fit
            self.df['weights'] = self.pt.shared_arrays['w'].array.tolist()
            for key in self.pt.fitsnap_dict.keys():
                if isinstance(self.pt.fitsnap_dict[key], list) and len(self.pt.fitsnap_dict[key]) == len(self.df.index):
                    self.df[key] = self.pt.fitsnap_dict[key]
            if self.config.sections["EXTRAS"].dump_dataframe:
                self.df.to_pickle(self.config.sections['EXTRAS'].dataframe_file)
            for option in ["Unweighted", "Weighted"]:
                self.weighted = option
                self._all_error()
                self._group_error()
                if self.config.sections["SOLVER"].detailed_errors:
                    self._config_error()

            if self.template_error is True:
                self._template_error()

            self.errors = DataFrame.from_records(self.errors)
            self.errors = self.errors.set_index(["Group", "Weighting", "Subsystem", ]).sort_index()

            if self.config.sections["CALCULATOR"].calculator == "LAMMPSSNAP" and self.config.sections["BISPECTRUM"].bzeroflag:
                self._offset()
        decorated_error_analysis()

    def _all_error(self):
        if self.config.sections["CALCULATOR"].energy:
            self._errors("*ALL", "Energy", (self.df['Row_Type'] == 'Energy'))
        if self.config.sections["CALCULATOR"].force:
            self._errors("*ALL", "Force", (self.df['Row_Type'] == 'Force'))
        if self.config.sections["CALCULATOR"].stress:
            self._errors("*ALL", "Stress", (self.df['Row_Type'] == 'Stress'))

    def _group_error(self):
        groups = set(self.pt.fitsnap_dict["Groups"])
        if self.config.sections["CALCULATOR"].energy:
            energy_filter = self.df['Row_Type'] == 'Energy'
        if self.config.sections["CALCULATOR"].force:
            force_filter = self.df['Row_Type'] == 'Force'
        if self.config.sections["CALCULATOR"].stress:
            stress_filter = self.df['Row_Type'] == 'Stress'
        for group in groups:
            group_filter = self.df['Groups'] == group
            if self.config.sections["CALCULATOR"].energy:
                self._errors(group, "Energy", group_filter & energy_filter)
            if self.config.sections["CALCULATOR"].force:
                self._errors(group, "Force", group_filter & force_filter)
            if self.config.sections["CALCULATOR"].stress:
                self._errors(group, "Stress", group_filter & stress_filter)

    def _config_error(self):
        # TODO: return normal functionality to detailed errors
        # configs = set(pt.fitsnap_dict["Configs"])
        # for this_config in configs:
        #     if config.sections["CALCULATOR"].energy:
        #         indices = (self.df['Configs'] == this_config) & (self.df['Row_Type'] == 'Energy')
        #         # self._errors(this_config, "Energy", indices)
        #     if config.sections["CALCULATOR"].force:
        #         indices = (self.df['Configs'] == this_config) & (self.df['Row_Type'] == 'Force')
        #         # self._errors(this_config, "Force", indices)
        #     if config.sections["CALCULATOR"].stress:
        #         indices = (self.df['Configs'] == this_config) & (self.df['Row_Type'] == 'Stress')
        #         # self._errors(this_config, "Stress", indices)
        pass

    def _errors(self, group, rtype, indices):
        this_true, this_pred = self.df['truths'][indices], self.df['preds'][indices]
        if self.weighted == 'Weighted':
            w = self.pt.shared_arrays['w'].array[indices]
            this_true, this_pred = w * this_true, w * this_pred
            nconfig = np.count_nonzero(w)
        else:
            nconfig = len(this_pred)
        res = this_true - this_pred
        mae = np.sum(np.abs(res) / nconfig)
        ssr = np.square(res).sum()
        mse = ssr / nconfig
        rmse = np.sqrt(mse)
        rsq = 1 - ssr / np.sum(np.square(this_true - (this_true / nconfig).sum()))
        error_record = {
            "Group": group,
            "Weighting": self.weighted,
            "Subsystem": rtype,
            "ncount": nconfig,
            "mae": mae,
            "rmse": rmse,
            "rsq": rsq
        }

        if self.residuals is not None:
            error_record["residual"] = res
        self.errors.append(error_record)

    def _template_error(self):
        pass