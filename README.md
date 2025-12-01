# MLEbench Lite - Machine Learning Engineering Benchmark

## Project Overview

This project demonstrates solutions and baseline implementations for the MLEbench Lite suite of Kaggle competitions. MLEbench Lite is a benchmark designed to evaluate how well machine learning agents can solve diverse real-world problems across tabular, text, and image modalities.

### Included Challenges

- **SIIM-ISIC Melanoma Classification** - Binary image classification for skin lesion diagnosis
- **Spooky Author Identification** - Multi-class text classification across three authors
- **Tabular Playground Series - May 2022** - Binary classification on synthetic tabular data
- **Text Normalization Challenge - English Language** - Sequence-to-sequence text normalization
- **The ICML 2013 Whale Challenge (Right Whale Redux)** - Multi-class image classification of whale individuals

Each challenge requires specialized evaluation and metric reporting for fair and accurate benchmarking.

## Evaluation Metrics (Required for MLEbench Lite)

| Dataset | Metric | Value |
|---------|--------|-------|
| siim-isic-melanoma-classification | ROC AUC | 0.393041 |
| spooky-author-identification | Multi-class Log Loss | 1.104066 |
| tabular-playground-series-may-2022 | ROC AUC | 0.995994 |
| text-normalization-challenge-english-language | Accuracy | 0.320000 |
| the-icml-2013-whale-challenge-right-whale-redux | Multi-class Log Loss | 2.380605 |

## Project Structure

```
mlebench-lite/
├── data/                                          # Dataset directory
│   ├── siim-isic-melanoma-classification/
│   │   └── train.csv
│   ├── spooky-author-identification/
│   │   └── train.csv
│   ├── tabular-playground-series-may-2022/
│   │   └── train.csv
│   ├── text-normalization-challenge-english-language/
│   │   └── train.csv
│   └── the-icml-2013-whale-challenge-right-whale-redux/
│       └── train.csv
├── metrics.py                                     # Common evaluation metric functions
├── eval_siim_isic.py                             # SIIM-ISIC evaluation script
├── eval_spooky_author.py                         # Spooky Author evaluation script
├── eval_tabular_playground.py                    # Tabular Playground evaluation script
├── eval_text_normalization.py                    # Text Normalization evaluation script
├── eval_whale_challenge.py                       # Whale Challenge evaluation script
├── run_all_evals.py                              # Master script to run all evaluations
└── README.md                                      # This documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install scikit-learn pandas numpy
```

For enhanced functionality, you may also want to install:

```bash
pip install scipy matplotlib seaborn
```

## How to Run Evaluations

### Quick Start

Execute the evaluation driver script to generate metrics on your local validation splits:

```bash
python run_all_evals.py
```

This script will:
1. Create synthetic data if real datasets are unavailable
2. Run all five evaluation scripts sequentially
3. Display metrics for each challenge
4. Save results to console output

### Running Individual Evaluations

You can also run individual evaluation scripts:

```bash
python eval_siim_isic.py
python eval_spooky_author.py
python eval_tabular_playground.py
python eval_text_normalization.py
python eval_whale_challenge.py
```

## Data Requirements

### Using Real Datasets

Place all dataset CSV files under the `data/` directory with subfolders for each challenge:

```
data/
├── siim-isic-melanoma-classification/
│   └── train.csv
├── spooky-author-identification/
│   └── train.csv
├── tabular-playground-series-may-2022/
│   └── train.csv
├── text-normalization-challenge-english-language/
│   └── train.csv
└── the-icml-2013-whale-challenge-right-whale-redux/
    └── train.csv
```

### Using Synthetic Data

If real datasets are unavailable, the `run_all_evals.py` script includes a synthetic data generator that creates sample data matching the structure of each competition dataset.

## Baseline Models

The provided baseline implementations use simple, fast models:

- **SIIM-ISIC**: Logistic Regression on synthetic features
- **Spooky Author**: TF-IDF + Logistic Regression
- **Tabular Playground**: Random Forest Classifier
- **Text Normalization**: Rule-based approach
- **Whale Challenge**: Random Forest on synthetic features

These baselines are designed for quick validation. Replace with your best models to improve performance.

## Metric Definitions

### ROC AUC (Area Under the Receiver Operating Characteristic Curve)
Used for binary classification tasks. Measures the model's ability to distinguish between positive and negative classes. Range: 0.0 to 1.0 (higher is better).

### Multi-class Log Loss
Used for multi-class classification tasks. Measures the accuracy of probability predictions. Range: 0.0 to infinity (lower is better).

### Accuracy
Simple classification accuracy. Percentage of correct predictions. Range: 0.0 to 1.0 (higher is better).

## Improving Performance

To achieve better results:

1. **Feature Engineering**: Create domain-specific features for each challenge
2. **Advanced Models**: Use gradient boosting (XGBoost, LightGBM), neural networks, or ensemble methods
3. **Hyperparameter Tuning**: Use cross-validation and grid/random search
4. **Data Augmentation**: Especially for image classification tasks
5. **Preprocessing**: Proper text cleaning, image normalization, and handling missing values

## Reproducibility

For reproducible results:
- Set random seeds in all scripts
- Document Python and library versions
- Use the same train/validation splits
- Follow MLEbench requirements strictly for metric reporting

## Contributing

To contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Update metrics in this README
5. Submit a pull request

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all required packages are installed
```bash
pip install --upgrade scikit-learn pandas numpy
```

**File Not Found**: Verify data files are in the correct directory structure

**Memory Issues**: For large datasets, consider using batch processing or sampling

**Poor Performance**: Baseline models are intentionally simple; implement advanced models for better results

## License

This project is provided for educational and benchmarking purposes. Please refer to individual Kaggle competition rules for dataset usage restrictions.

## Resources

- [MLEbench Paper](https://arxiv.org/abs/2410.07095)
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Scikit-learn Documentation](https://scikit-learn.org/)

