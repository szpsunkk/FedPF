# Accurate Target Privacy Preserving Federated Learning Balancing Fairness and Utility

This paper proposed a private and fair federated learning algorithm (FedPF) with protected sensitive dataset. We study the relationship between fairness, privacy and utility in Federated Learning.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation


```bash
# code
git clone https://github.com/szpsunkk/FedPF.git
cd FedPF
pip install -r requirements.txt
```

## Usage

```bash
# Jupyter templete

./system/federated_exponential_mechanism_adult.ipynb

# or the python code
python system/main.py

```

The comparison with baseline algorithms:

```
./system/compare_adult.ipynb
```

## Results
We consider three scenerios, including fairness metrics, privacy metrics, fairness and privacy metrics.

The templete is in the file `system/federated_exponential_mechanism_adult.ipynb` and `system/federated_exponential_mechanism_bank.ipynb`

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
