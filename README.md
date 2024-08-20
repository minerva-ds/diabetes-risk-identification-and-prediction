# D.R.I.P.: Diabetes Risk Identification and Prediction Tool

![Diabetes Risk Prediction](images/drip-header.png)

## Business Understanding

## Data Understanding

## Modeling and Evaluation

## Conclusion

## Main Project Files
[**D.R.I.P.**](https://minerva-ds.github.io/diabetes-risk-identification-and-prediction/deployment/) 
<br>Usable single page application that uses your self reported data to predict your diabetes risk.

[**Dashboard**](https://minerva-ds.github.io/diabetes-risk-identification-and-prediction/dashboard_files/combined_dashboard.html)
<br>Interactive charts from the project.

[**Final Notebook**](notebook.ipynb)
<br>The entire process used to make the tool and charts well documented in a Jupyer notebook with markdown.

[**Presentation**](presentation.pdf)
<br>The presentation slides.

## Further Development

## Repository Structure

| Directory/File                     | Description                                               |
|------------------------------------|-----------------------------------------------------------|
| **/data**                          | Data files used in analysis                               |
| ├── cdc_diabetes_data_features.pkl | Pickled file containing feature data                      |
| ├── cdc_diabetes_data_metadata.json| Metadata for the dataset                                  |
| ├── cdc_diabetes_data_targets.pkl  | Pickled file containing target data                       |
| ├── cdc_diabetes_data_variables.pkl| Information about dataset variables                       |
| **/docs**                          | Root directory for GitHub Pages content                   |
| ├── **/dashboard_files**           | Directory for dashboard-related HTML files                |
| │   ├── combined_dashboard.html    | Combined view of all analysis dashboards                  |
| │   ├── mean_impact.html           | Dashboard showing mean impact of variables                |
| │   ├── roc_auc.html               | ROC AUC curve for the models                              |
| │   └── threshold_analysis.html    | Analysis of various threshold settings                    |
| ├── **/deployment**                | Files related to model deployment                         |
| │   ├── catboost_model.js          | JavaScript file for model deployment                      |
| │   ├── index.html                 | Main HTML file for the deployment site                    |
| │   ├── **/models**                | Stored models in various formats                          |
| │   │   ├── catboost_model_fixed.onnx | ONNX model with fixed parameters                        |
| │   │   ├── catboost_model_no_zipmap.onnx | ONNX model without zipmap                             |
| │   │   ├── catboost_model.onnx    | Default CatBoost model in ONNX format                     |
| │   │   └── model.onnx             | General model file                                        |
| │   ├── **/scripts**               | Scripts used in deployment                                |
| │   │   └── dump_model.py          | Script for dumping model info for debugging
| │   ├── styles.css                 | CSS styles for the deployment frontend                    |
| │   └── **/test_onnx**             | HTML files for testing ONNX models                        |
| │       └── simple-example.html    | Simple HTML file for ONNX model testing                   |
| └── index.html                     | Main HTML file linking to project components              |
| **/environment_no_builds.yml**     | More system agnostic conda environment file                                    |
| **/environment_ubuntu.yml**        | Ubuntu-specific Conda environment file                    |
| **/images**                        | Directory for storing images used in README or notebooks  |
| ├── drip-header.png                | Header image for README or documentation                  |
| **/notebook.ipynb**                | Jupyter notebook with the project analysis                |
| **/README.md**                     | README file for project overview and navigation           |
| **/scrapbook**                     | Additional notebooks for exploratory analysis             |
| └── scrapbook.ipynb                | Notebook for storing miscellaneous analyses               |


## Reproducibility
### Prerequisites
Ensure you have these before continuing with the rest of the setup!

#### 1. **Clone the Repository**
It's assumed you have Git installed and are familiar with basic Git commands. If you haven't yet installed Git, you can download it from [git-scm.com](https://git-scm.com).

To clone the repository, open your command line tool:
- **macOS/Linux**: Open Terminal.
- **Windows**: Open Git Bash (recommended for Git operations).

Then execute the following commands:
```bash
git clone https://github.com/minerva-ds/diabetes-risk-identification-and-prediction
cd diabetes-risk-identification-and-prediction
```

#### 2. **Install Conda**
Ensure Conda is installed on your system. If not already installed, you can download it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).

### Quick Setup Options
After completing the prerequisites, choose one of the following quick setup options based on your operating system.

#### For Ubuntu Users
Quickly set up the environment using the `environment_ubuntu.yml` provided in the repository:
```bash
conda env create -f environment_ubuntu.yml
conda activate drip_env
```

#### For Non-Ubuntu Users
Use the `environment_no_builds.yml` for a smoother setup across different systems:
```bash
conda env create -f environment_no_builds.yml
conda activate drip_env
```
**Note:** The `--no-builds` option is used to enhance compatibility, but it's not guaranteed to work in all cases. If you encounter any issues, consider the manual setup instructions below.

### Manual Installation Steps
If the quick setup options do not meet your needs or you encounter issues, follow these detailed manual installation steps:

#### 1. **Create and Activate Environment**
Create and activate the Conda environment:
```bash
conda create -n drip_env python=3.10 pandas scipy scikit-learn seaborn matplotlib ipython sweetviz xgboost catboost pytorch-tabnet optuna onnx onnxruntime plotly -c conda-forge -y
conda activate drip_env
```

#### 2. **Install Additional Packages**
After activating the environment, proceed to install these additional packages:
```bash
conda install ipykernel -y
conda install skl2onnx -y
pip install ucimlrepo
```

#### 3. **Verify Installation**
Finally, to ensure all packages are installed correctly, run:
```bash
conda list
```

This setup process ensures your development environment is prepared correctly, letting you focus fully on your project.