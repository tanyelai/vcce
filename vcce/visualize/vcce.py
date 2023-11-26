import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

class VCCE:
    """
    Visualizing Counterfactual Clues for 12-lead ECG signals
    """

    def __init__(self, signal, extracted_features=None, counterfactuals=None, feature_count=None):
        self.signal = signal
        self.extracted_features = extracted_features
        self.counterfactuals = counterfactuals
        self.feature_count = feature_count
        self.visualization_graph = None
        self.feature_list = ['P', 'Q', 'R', 'S', 'T', 'ST', 'PT', 'RS', 'QS']
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL',
                           'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def prepare_data(self):
        """
        Merge the counterfactuals with the extracted features.

        Returns:
            pandas.DataFrame: A merged dataframe containing counterfactuals and extracted features.
        """
        return self.counterfactuals.merge(self.extracted_features[["Beat_R_idx",
                                                                   "Beat_P_idx",
                                                                   "Beat_Q_idx",
                                                                   "Beat_S_idx",
                                                                   "Beat_T_idx"]],
                                          on="Beat_R_idx",
                                          how="inner")

    def create_visualization_graph(self, show_available_features=False, show_graph=False):
        self.counterfactuals = self.prepare_data()

        # Filter the DataFrame to get only values above 0
        filtered_df = self.counterfactuals[self.counterfactuals.iloc[:, 1:(self.feature_count+1)] > 0]

        # Iterate through each row and concatenate column names
        result = []
        for index, row in filtered_df.iterrows():
            column_names = row.index[row.notnull()].tolist()
            # Filtering out columns with '_mean' in their names
            column_names = [col for col in column_names if '_mean' not in col]
            result.append(column_names)

        if show_available_features:
            for row in result:
                print(row)

        visualization_df = pd.DataFrame(columns=['lead'], data=self.lead_names)

        for feature in self.feature_list:
            visualization_df[feature] = None

        for j in range(len(result)):
            for i in range(len(result[j])):
                lead = result[j][i].split('_')[0]
                feature = result[j][i].split('_')[1]

                if feature.endswith("mean"):  # ignoring _mean features
                    continue
                
                if feature in self.feature_list:
                    condition = visualization_df['lead'] == lead
                    
                    indices = self.counterfactuals[f'Beat_{feature}_idx'][j] if "idx" in feature else self.counterfactuals[f'Beat_{feature[0]}_idx'][j]
                    
                    if not isinstance(indices, list):  # Ensuring indices is a list
                        indices = [indices]
                        
                    existing_val = visualization_df.loc[condition, feature].values[0]

                    value = filtered_df.loc[j, f'{lead}_{feature}']
                    new_val_dict = {idx: value for idx in indices}
                    
                    if existing_val is not None and isinstance(existing_val, dict):
                        # Merging dictionaries and summing up the counts for the common keys
                        existing_val = {k: new_val_dict.get(k, 0) + existing_val.get(k, 0) for k in set(new_val_dict) | set(existing_val)}
                        visualization_df.loc[condition, feature] = [existing_val]  # Wrapping in a list
                    else:
                        visualization_df.loc[condition, feature] = [new_val_dict]  # Wrapping in a list


        if show_graph:
            print("RAW DATAFRAME WITH COUNTERFACTUAL COUNTS")
            print(filtered_df)

            print("\nVISUALIZATION GRAPH WITH COUNTERFACTUAL COUNTS")
            print(visualization_df)

        self.visualization_graph = visualization_df

        return visualization_df


    def extract_peaks(self, peak_type, lead_index):
        """
        Args:
            peak_type (str): Type of the peak.
            lead_index (int): Index of the lead.

        Returns:
            array or list of tuples: Array of peak indices for 'P', 'Q', 'R', 'S', 'T' peaks.
                                        List of tuples of peak indices for 'ST', 'PT', 'RS', 'QS' peaks.
        """
        if self.visualization_graph is None:
            self.create_visualization_graph()

        peaks = self.visualization_graph.loc[lead_index, peak_type]

        if peaks is None or not isinstance(peaks, dict):
            return []

        # Convert dict to list of tuples (index, power)
        peaks = [(index, power) for index, power in peaks.items()]
        
        if peak_type in ['ST', 'PT', 'RS', 'QS']:
            pairs = []
            for f_idx, power in peaks:
                s_indices = self.counterfactuals[f'Beat_{peak_type[1]}_idx']
                if s_indices is None or len(s_indices) == 0:
                    continue
                corresponding_s_idx = s_indices[self.counterfactuals[
                    f'Beat_{peak_type[0]}_idx'] == f_idx].values[0]
                pairs.append(((f_idx, corresponding_s_idx), power))
            return pairs
        else:
            return peaks

    def plot_peaks(self, ax, peaks, signal, color, label, zorder):
        """
        Plots the peaks of a signal on a given axis.

        Args:
        ax (matplotlib.axes.Axes): The axis to plot the peaks on.
        peaks (numpy.ndarray): An array of peak indices.
        signal (numpy.ndarray): The signal to plot the peaks of.
        color (str): The color of the plotted peaks.
        label (str): The label of the plotted peaks.
        zorder (int): The z-order of the plotted peaks.

        Returns:
        numpy.ndarray: An array of peak indices.
        """
        # Extract the indices from peaks_with_power
        indices = [index for index, power in peaks]
        
        ax.scatter(
            indices,
            signal[indices],
            color=color,
            label=label,
            zorder=zorder
        )
    
        return indices

 

    def plot_dotted_lines(self, ax, indices_with_power, signal, color):
        for (idx1, idx2), power in indices_with_power:
            amplitude1 = signal[idx1]
            amplitude2 = signal[idx2]
            
            # Determine the lower amplitude point (S) and higher amplitude point (T)
            lower_idx, higher_idx = (idx1, idx2) if amplitude1 < amplitude2 else (idx2, idx1)
            lower_amplitude, higher_amplitude = signal[lower_idx], signal[higher_idx]
            
            # Set a small offset below the lower amplitude point
            offset = 0.1 * (higher_amplitude - lower_amplitude)
            
            # Draw a faint horizontal line a little below the lower point to the higher point
            ax.hlines(y=lower_amplitude - offset, xmin=lower_idx, xmax=higher_idx, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            
            # Draw a colored vertical line from the higher point to the horizontal line
            ax.vlines(x=higher_idx, ymin=lower_amplitude - offset, ymax=higher_amplitude, color=color, linewidth=2, alpha=0.8)

            # Calculate the amplitude difference and annotate it below the grey line
            amplitude_diff = abs(amplitude1 - amplitude2)
            annotation_y = lower_amplitude - 2 * offset  # Adjust this as necessary
            annotation_x = (lower_idx + higher_idx) / 2  # Midpoint of lower_idx and higher_idx
            ax.text(annotation_x, annotation_y, f"{amplitude_diff:.2f}", 
                    ha='center', va='center', color='black', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='white', boxstyle='round,pad=0.5'))



    def plot_emphasis(self, ax, peaks, signal, color, label=None, zorder=None):
        """
        Plots a shaded region around each peak in the signal.

        Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the emphasis.
        peaks (list): A list of indices representing the peaks in the signal.
        signal (numpy.ndarray): The signal to plot.
        color (str): The color of the shaded region.
        label (str, optional): The label for the shaded region. Defaults to None.
        zorder (int, optional): The z-order of the shaded region. Defaults to None.
        """
        max_alpha = 0.3
    
        # extracting the power values and finding the max power
        power_values = [power for _, power in peaks]
        max_power = max(power_values) if power_values else 1  # avoid division by zero
        
        for peak_info in peaks:
            if isinstance(peak_info, tuple):
                peak_sample, power = peak_info
                alpha = (power / max_power) * max_alpha
            else:
                peak_sample, alpha = peak_info, max_alpha  # default alpha
            
            if alpha > max_alpha:
                alpha = max_alpha  # ensure that alpha doesn't exceed the maximum permissible value
                
            if peak_sample - 6 >= 0 and peak_sample + 6 < len(signal):
                ax.axvspan(peak_sample - 6, peak_sample + 6, color=color, alpha=alpha, label=label, zorder=zorder)

                
    def plot_legend(self, ax, handles, labels):
        ax.legend(handles, labels, loc='upper left', ncol=1,
                  fontsize=18, facecolor='white', framealpha=0.95)

    def plot_ecg_report(self,
                        signal=None,
                        sampling_rate=100,
                        method="koka2022",
                        show_peaks=False,
                        show_advanced=False,
                        show_emphasis=True,
                        fig_size=(32, 26), fig_col=2,
                        save=False, 
                        save_format="pdf", save_path=".", 
                        file_name="vcce_report"):
        """
        Plot the ECG report for the counterfactual clues

        Args:
            signal (array, optional): ECG signal. Defaults to None.
            sampling_rate (int, optional): Sampling rate of the ECG signal. Defaults to 100.
            method (str, optional): Method to process the ECG signal. Defaults to "koka2022".
            show_peaks (bool, optional): Show the peaks in the ECG signal. Defaults to False.
            show_advanced (bool, optional): Show the features computed via amplitude differences as dotted lines overlaying the ECG signal. Defaults to False.
            show_emphasis (bool, optional): Show the peaks in the ECG signal as shaded regions. Defaults to True.
            fig_size (tuple, optional): Size of the figure. Defaults to (32, 26).
            fig_col (int, optional): Number of columns in the figure. Defaults to 2.
                Note: If column size is 1, the recommended fig_size is (32, 48).
            save (bool, optional): Save the ECG report. Defaults to False.
            save_format (str, optional): Format of the saved ECG report. Defaults to "pdf".
            save_path (str, optional): Path to save the ECG report. Defaults to ".".
            file_name (str, optional): Name of the saved ECG report. Defaults to "vcce_report".
        """

        num_rows = len(self.lead_names) // fig_col
        num_cols = fig_col

        fig, axs = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=fig_size, constrained_layout=True)

        legend_handles = []
        legend_labels = []

        for lead_index, lead in enumerate(self.lead_names):
            row_index = lead_index // num_cols
            col_index = lead_index % num_cols
            ax = axs[row_index, col_index] if num_cols > 1 else axs[row_index]
            
            ax.set_ylabel(f"{lead}", fontsize=18)
            ax.set_facecolor('#EFEFEF')
            ax.grid(color='darkred', linestyle='-.', linewidth=0.5)

            if show_peaks or show_advanced or show_emphasis:
                signal_, _ = nk.ecg_process(signal[:, lead_index], sampling_rate=sampling_rate, method=method)
            else:
                signal_ = pd.DataFrame({"ECG_Raw": signal[:, lead_index]})
            
            ax.plot(signal_["ECG_Raw"], color="black", label="Raw", linewidth=2)
            
            if "Raw" not in legend_labels:
                legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=2))
                legend_labels.append("Raw")

            if show_peaks:
                peak_types = [('P', 'blue', 'P-peaks'), ('Q', 'green', 'Q-peaks'), ('R', '#FFC107', 'R-peaks'), ('S', 'red', 'S-peaks'), ('T', 'purple', 'T-peaks')]
            
                for peak_type, color, label in peak_types:
                    peaks = self.plot_peaks(ax, self.extract_peaks(peak_type, lead_index), signal_["ECG_Raw"], color, label, 1)
                    if peaks and label not in legend_labels:
                        legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, linestyle='None', label=label))
                        legend_labels.append(label)

            if show_advanced:
                advanced_types = [('ST', 'red'), ('PT', 'blue'), ('RS', '#FFA000'), ('QS', 'green')]

                for advanced_type, color in advanced_types:
                    if self.extract_peaks(advanced_type, lead_index):
                        self.plot_dotted_lines(ax, self.extract_peaks(advanced_type, lead_index), signal_["ECG_Raw"], color)
                        if advanced_type not in legend_labels:  # Checking if label is already in legend_labels
                            legend_handles.append(plt.Line2D([0], [0], linestyle="-", color=color, label=advanced_type, linewidth=2))
                            legend_labels.append(advanced_type)

            if show_emphasis:
                for peak_type, color in [('P', 'blue'), ('Q', 'green'), ('R', '#FFA000'), ('S', 'red'), ('T', 'purple')]:
                    peaks = self.extract_peaks(peak_type, lead_index)
                    if peaks:
                        self.plot_emphasis(ax, peaks, signal_["ECG_Raw"], color)
                        if peak_type not in legend_labels:
                            legend_handles.append(plt.Line2D([0], [0], linestyle="-", color=color, linewidth=6, alpha=0.5))
                            legend_labels.append(peak_type)

        #plt.suptitle("ECG Signal Report", fontsize=18, y=1.005)

        #plt.savefig(f"{method}_report.png", dpi=300)

        # When adding the aggregate legend
        if legend_handles and legend_labels:
            ncol = 2 if len(legend_handles) > 8 else 1  # if the number of legend items is more than 8, then ncol will be 2, else 1.
            # If axs is 2D
            if axs.ndim == 2:
                axs[0, 0].legend(legend_handles, legend_labels, loc='upper left', fontsize=18, facecolor='white', framealpha=0.95, ncol=ncol)
            # If axs is 1D
            else:
                axs[0].legend(legend_handles, legend_labels, loc='upper left', fontsize=18, facecolor='white', framealpha=0.95, ncol=ncol)
        
        if save:
            os.makedirs(save_path, exist_ok=True)
            if save_format not in {"pdf", "png", "svg"}:
                print("Invalid save type. Please choose from 'pdf', 'png' or 'svg'")
            else:
                plt.savefig(f"{save_path}/{file_name}.{save_format}", dpi=144 if save_format == "png" else None)
        
        plt.show()
