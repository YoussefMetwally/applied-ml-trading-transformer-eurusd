# Applied ML Trading: Transformer for EURUSD Forecasting 📈

![GitHub Release](https://img.shields.io/github/release/YoussefMetwally/applied-ml-trading-transformer-eurusd.svg)

Welcome to the **Applied ML Trading** repository! This project focuses on developing and fine-tuning a **TimeSeriesTransformer** model to forecast the 5-minute closing prices of the EURUSD currency pair. This modern approach serves as a counterpart to a baseline LSTM model, providing insights into the dynamics of forex trading.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In the fast-paced world of algorithmic trading, accurate forecasting models can make a significant difference. This project aims to harness the power of deep learning through transformer models to predict price movements in the EURUSD market. The TimeSeriesTransformer leverages self-attention mechanisms, which allow it to focus on relevant past price points while making predictions.

## Project Overview

The primary goal of this repository is to create a robust model that can predict EURUSD closing prices based on historical data. By comparing the transformer model's performance with that of an LSTM baseline, we aim to showcase the effectiveness of modern deep learning techniques in financial forecasting.

## Technologies Used

This project incorporates various technologies and libraries:

- **Python**: The primary programming language.
- **PyTorch**: For building and training the deep learning models.
- **Hugging Face Transformers**: For utilizing transformer architectures.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-learn**: For preprocessing and evaluation metrics.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YoussefMetwally/applied-ml-trading-transformer-eurusd.git
   cd applied-ml-trading-transformer-eurusd
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installing the necessary packages, you can start using the model. The repository contains scripts for data preprocessing, model training, and evaluation.

### Data Preprocessing

Prepare your dataset by running the following script:

```bash
python preprocess.py
```

This script will load historical EURUSD data, clean it, and prepare it for training.

### Training the Model

To train the TimeSeriesTransformer model, execute:

```bash
python train.py
```

You can adjust hyperparameters in the `config.yaml` file to optimize performance.

### Making Predictions

Once the model is trained, you can use it to make predictions on new data:

```bash
python predict.py
```

This script will output predicted closing prices for the specified time frame.

## Model Training

The training process involves several steps:

1. **Data Loading**: Load the historical data for EURUSD.
2. **Feature Engineering**: Create features that will help the model learn.
3. **Model Definition**: Define the TimeSeriesTransformer architecture.
4. **Training Loop**: Train the model over several epochs, adjusting weights based on loss.
5. **Evaluation**: Assess the model's performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Evaluation Metrics

To evaluate the model's performance, we use the following metrics:

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **Root Mean Squared Error (RMSE)**: Provides a measure of how far predictions deviate from actual values.

## Results

The results of the model's performance will be documented in the `results/` directory. You can visualize the predictions against actual prices using the provided plotting scripts.

![Model Predictions](https://img.shields.io/badge/Model%20Predictions-Results-blue)

## Contributing

We welcome contributions to improve this project. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or suggestions, please reach out to me at [YoussefMetwally](https://github.com/YoussefMetwally).

For releases and updates, check out the [Releases section](https://github.com/YoussefMetwally/applied-ml-trading-transformer-eurusd/releases).

---

Thank you for visiting the **Applied ML Trading** repository! Your interest in modern financial forecasting is greatly appreciated.