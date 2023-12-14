# VCCE

This source code is associated with the paper 'Interpretable ECG Analysis for Myocardial Infarction Detection through Counterfactuals'. It is designed for data management specific to PTB-XL, along with processes for feature extraction, selection, and the extraction of counterfactual clues. Additionally, the code includes a unique application for visualizing these counterfactual clues on electrocardiogram data.


## Installation

**To install this project, follow these steps:**

1. Clone the repository to your local machine using the following command:

```
git clone https://github.com/tanyelai/vcce.git
```

2. Change to the project directory:

```
cd vcce
```

3. Install the dependencies:

```
pip install -r requirements.txt
```

4. Use **Example Jupyter Notebooks** to start

We have provided example Jupyter notebooks in the notebooks folder to help you understand some of the functionalities in the library. To run these notebooks, you will need to have Jupyter Notebook installed on your machine. VSCode is also recommended.

We encourage you to explore the notebooks and experiment with the code to get a better understanding.

## Data Folder

To use the notebooks for data processing, please follow these steps:

1. Download the dataset from [https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/) and place it in data directory.

2. Run the notebooks in the following order for data management, feature extraction, feature selection, and counterfactual explanation (cfe) processing. This sequence will allow you to generate the necessary study files for replication:

   - `data_management.ipynb`
   - `feature_extraction.ipynb`
   - `select_features.ipynb`
   - `cfe_process.ipynb`

Please feel free to explore and adapt these notebooks and source codes to align with your specific research requirements. If you encounter any problems or have questions, don't hesitate to contact the research team for assistance.

## Citation
@misc{tanyel2023interpretable,
      title={Interpretable ECG Analysis for Myocardial Infarction Detection through Counterfactuals}, 
      author={Toygar Tanyel and Sezgin Atmaca and Kaan Gökçe and M. Yiğit Balık and Arda Güler and Emre Aslanger and İlkay Öksüz},
      year={2023},
      eprint={2312.08304},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
