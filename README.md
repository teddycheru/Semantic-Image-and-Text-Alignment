# Automated Storyboard Synthesis for Digital Advertising

This project aims to streamline the workflow for creating storyboard ads by utilizing a combination of data preprocessing, image analysis, and text analysis.

## Repository structure
```plaintext
Semantic-Image-and-Text-Alignment
│
├── notebooks
│   ├── assets_eda.ipynb
│   ├── concepts_eda.ipynb
│   ├── data_analysis.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_evaluation.ipynb
│   └── performance_eda.ipynb
│
├── results
│   └── figures
|     └── README.md
│
├── Scripts
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── model_evaluation.py
│   └── image_segmentation.py
│
├──tests
│   ├── test_data_preprocessing.ipynb
│   ├── test_imaage_segmentation.py
│   ├── test_model_evaluation.ipynb
│   └── README.md
│
├──.gitignore
│
├── README.md
│
└── requirements.txt
```
## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/teddycheru/Semantic-Image-and-Text-Alignment.git
    cd Semantic-Image-and-Text-Alignment
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Explore the data and workflow strategy using the Jupyter notebooks in the `notebooks/` directory.
2. Run the scripts located in the `scripts/` directory for various tasks such as data preprocessing, image analysis, text analysis, and image composition.
3. Generate reports and figures by executing the scripts and notebooks.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
