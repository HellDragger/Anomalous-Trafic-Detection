# Anomaly Detection in Network Traffic

This repository contains Jupyter notebooks for detecting anomalies in network traffic using Recurrent Neural Networks (RNN) and Deep Q-Learning (DQL). The notebooks are designed to provide practical implementations of these machine learning techniques to analyze and detect anomalies in the NF-UNSW-NB15 dataset.

## Table of Contents

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Notebooks](#notebooks)
  - [rnn-anomaly.ipynb](#rnn-anomalyipynb)
  - [nn-dql.ipynb](#nn-dqlipynb)
- [Dataset](#dataset)
- [License](#license)

## Requirements

To run the notebooks, you will need the following software:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Necessary libraries (you can install them using `pip`):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

## Getting Started

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/anomaly-detection.git
   cd anomaly-detection
   ```

2. Ensure that you have the required dataset downloaded (see the Dataset section below).

3. Open the notebooks in Jupyter:

   ```bash
   jupyter notebook
   ```

4. Run the notebooks sequentially to execute the anomaly detection algorithms.

## Notebooks

### rnn-anomaly.ipynb

This notebook implements an RNN model to detect anomalies in network traffic data. It includes:

- Data preprocessing steps.
- RNN model architecture and training.
- Evaluation metrics to assess the model's performance.

### nn-dql.ipynb

This notebook implements a Deep Q-Learning algorithm for anomaly detection. It includes:

- Environment setup for the reinforcement learning model.
- DQL agent architecture.
- Training and evaluation of the agent on the network traffic dataset.

## Dataset

The notebooks utilize the **NF-UNSW-NB15** dataset, which can be found in the following locations:

- [NF-UNSW-NB15-v2.csv]([https://kaggle.com/datasets/yourdatasetlink1](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA1))
- [NF-UNSW-NB15.csv]([https://kaggle.com/datasets/yourdatasetlink2](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA1))

Make sure to download the dataset and place it in the appropriate directory.

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [The University of Queensland, Australia](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA1) for providing the NF-UNSW-NB15 dataset.
