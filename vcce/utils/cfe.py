import numpy as np
import pandas as pd
import time
import os
import json

import dice_ml
from dice_ml import Dice

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression


class cfe_utils:
    def __init__(self, train=None, test=None, target=None, src=None, dest=None, path="./counterfactuals/explanations", classifier=XGBClassifier, random_state=42):
        self.train = train
        self.test = test
        self.target = target
        self.src = src
        self.dest = dest
        self.classifier = classifier
        self.random_state = random_state
        self.path = path
        self.label_mapping = {self.src: 0, self.dest: 1}


    def get_dice_trained_model(self):
            """
            Trains a classifier on the training data and returns a DiCE explainer object.

            Returns:
                DiCE explainer object.
            """
            self.train[self.target] = [self.label_mapping[label] for label in self.train[self.target]]

            x_train = self.train.drop(self.target, axis=1)
            y_train = self.train[self.target]
            
            clf = Pipeline(steps=[('preprocessor', StandardScaler()),
                                ('classifier', self._init_classifier())])
            model = clf.fit(x_train, y_train)

            column_list = x_train.columns

            data = x_train
            data["Label"] = y_train
            d = dice_ml.Data(dataframe=data, continuous_features= list(column_list), outcome_name='Label')
            m = dice_ml.Model(model=model, backend="sklearn")

            exp = Dice(d, m)
            
            return exp
    
    def _init_classifier(self):
        """
        Initializes the classifier.

        Returns:
            classifier object.
        This function can be further improved for custom classifiers.
        """

        if self.classifier == XGBClassifier:
            return XGBClassifier(use_label_encoder=False, eval_metric='error', random_state=self.random_state)
        else:
            return self.classifier(random_state=self.random_state)



    def get_counterfactuals(self, exp=None, cfe_count=3):

        """
        Generates counterfactuals for the test data and saves them to a JSON file.
        
        Args:
            exp (DiCE explainer object): DiCE explainer object.
            cfe_count (int, optional): Number of counterfactuals per beat. Defaults to 3.
        """

        #exp = self.get_dice_trained_model(self.train, self.test, self.src, self.classifier, random_state=42)

        time_data = []  # List to store the time data
        path = self.path

        for idx, row in self.test.iterrows():

            query = pd.DataFrame(columns=row.index, data=[row.values], index=[idx])

            start_time = time.time()  # Start the timer

            e1 = exp.generate_counterfactuals(query.drop(self.target, axis=1), total_CFs=cfe_count, desired_class="opposite", random_seed=42)

            end_time = time.time()  # End the timer
            cycle_time = end_time - start_time  # Calculate the time taken for this output cycle

            e1.visualize_as_dataframe(show_only_changes=True)

            json_str = e1.to_json()
            os.makedirs(f"{path}", exist_ok=True)
            jsonFile = open(f"{path}/beat_{idx}_{self.src}_to_{self.dest}.json", "w")
            jsonFile.write(json_str)
            jsonFile.close()

            # Append the time data to the list
            time_data.append([idx, self.src, self.dest, cycle_time])

        # Save time data to a text file
        with open(f"{path}/time_data.txt", "w") as txt_file:
            for data in time_data:
                txt_file.write(f"Beat's R idx: {data[0]}, Initial Diagnosis: {data[1]}, Final Diagnosis: {data[2]}, Time: {data[3]} seconds\n")

        # Save time data to an Excel file
        df_time_data = pd.DataFrame(time_data, columns=["Beat's R idx", "Initial Diagnosis", "Final Diagnosis", "Time (seconds)"])
        df_time_data.to_excel(f"{path}/time_data.xlsx", index=False)



    def json_to_excel(self):

        """
        Converts the JSON files to 3 different Excel files.
        One for the original data, one for the counterfactuals, and one for the merged data.
        """
        
        root_dir = self.path
        
        merged_data = pd.DataFrame()  # Initialize an empty DataFrame to merge the data
        merged_original = pd.DataFrame()  # Initialize an empty DataFrame to merge the data
        merged_cfes = pd.DataFrame()  # Initialize an empty DataFrame to merge the data

        path = f'{root_dir}/formatted_cfe_from_json'
        output_directory = f'{path}/{self.src}'
        os.makedirs(output_directory, exist_ok=True)
        
        # Iterate over the files in the directory
        for filename in os.listdir(root_dir):
            if filename.endswith(f'{self.src}_to_{self.dest}.json'):
                filepath = os.path.join(root_dir, filename)
                
                # Extract Rpeak index from filename
                rpeak_idx = filename.split('_')[1]

                # Read the JSON data from file
                with open(filepath, 'r') as file:
                    data = json.load(file)

                # Get original data
                original_df = pd.DataFrame(data['test_data'][0], columns=data['feature_names_including_target'])
                original_df['Beat_R_idx'] = int(rpeak_idx)

                # Get cfes
                cfe_df = pd.DataFrame(data['cfs_list'][0], columns=data['feature_names_including_target'])
                cfe_df['Beat_R_idx'] = int(rpeak_idx) 

                # Merge data into a single DataFrame
                merged_data = pd.concat([merged_data, original_df, cfe_df], ignore_index=True)
                merged_original = pd.concat([merged_original, original_df], ignore_index=True)
                merged_cfes = pd.concat([merged_cfes, cfe_df], ignore_index=True)

        # Save merged data as an Excel file
        merged_data.to_excel(os.path.join(output_directory, f"{self.src}_to_{self.dest}_merged_data.xlsx"), index=False)
        merged_original.to_excel(os.path.join(output_directory, f"{self.src}_to_{self.dest}_merged_original.xlsx"), index=False)
        merged_cfes.to_excel(os.path.join(output_directory, f"{self.src}_to_{self.dest}_merged_cfes.xlsx"), index=False)


    def find_change_count_in_excel(self, cfe_count, column_list):
        
        """
        Finds the number of feature changes in the counterfactuals and saves them to an Excel file.
        Args:
            cfe_count (int): Number of counterfactuals per beat.
            column_list (list): List of column names.
        """

        path = self.path
        src_path = f"{path}/formatted_cfe_from_json"
        dest_path = f"{path}/feature_changes"

        os.makedirs(os.path.join(dest_path, self.src), exist_ok=True)

        read_file = os.path.join(src_path, self.src, f'{self.src}_to_{self.dest}_merged_data.xlsx')
        write_file = os.path.join(dest_path, self.src, f'{self.src}_to_{self.dest}.xlsx')

        # Read the Excel file
        df = pd.read_excel(read_file, header=0)

        # Create a new column to track the baseline rows
        df['Baseline'] = False

        # Initialize the pattern count
        pattern_count = 0

        baseline_row = None

        # Initialize a flag to track whether the current baseline is valid
        valid_baseline = True

        # Create an empty dictionary to store feature change counts
        feature_changes = {}

        # Iterate over the rows
        for i, row in df.iterrows():
            if row[self.target] == self.label_mapping[self.src] and i % (cfe_count + 1) == 0:
                baseline_row = row
                df.at[i, 'Baseline'] = True
                valid_baseline = True  # Set the flag for a valid baseline
                pattern_count += 1
            elif i % (cfe_count + 1) == 0:
                baseline_row = row
                valid_baseline = False  # Set the flag for an invalid baseline
                df.at[i, 'Baseline'] = False  # Set the baseline flag to False
            else:  # Process the row only if the baseline is valid
                if valid_baseline:
                    rpeak_idx = int(row["Beat_R_idx"])
                    if rpeak_idx not in feature_changes:
                        feature_changes[rpeak_idx] = {column: 0 for column in column_list}

                    for column in column_list:
                        if column not in [self.target, "Baseline", "Beat_R_idx"]:
                            if row[column] == baseline_row[column]:
                                df.at[i, column] = '-'
                            if np.abs(row[column] - baseline_row[column]) < 0.01:
                                df.at[i, column] = '-'

                            if df.at[i, column] != '-':
                                feature_changes[rpeak_idx][column] += 1

        # Save the updated data to a new Excel file
        df.to_excel(write_file, index=False)

        # Create a new dataframe for feature change counts
        feature_changes_df = pd.DataFrame.from_dict(feature_changes, orient='index')

        feature_changes_df.index.name = 'Beat_R_idx'

        # Sort the dataframe by indices (Rpeak_idx)
        feature_changes_df.sort_index(inplace=True)

        #feature_changes_df = feature_changes_df.drop(columns=[self.target, 'Baseline', 'Beat_R_idx'])
        
        # Define the file path for the feature change counts Excel file
        feature_changes_file_path = os.path.join(dest_path, self.src, f'{self.src}_to_{self.dest}_feature_changes.xlsx')
        
        # Save the feature change counts dataframe to Excel
        feature_changes_df.to_excel(feature_changes_file_path)


    def get_column_list(self, column_string):
        """
        Convert the column string to a list.
        
        Args:
            column_string (str): A space-separated string of column names.
            
        Returns:
            list: A list of column names.
            
        Example:
            column_string = "Column1 Column2 Column3"
            column_list = get_column_list(column_string)
            # column_list will be ['Column1', 'Column2', 'Column3']
        """
        return column_string.split()