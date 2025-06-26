# Balancing the Tightrope: Privacy, Fairness, and Utility in Federated Learning

This paper proposed a private and fair federated learning algorithm (FedPF) with protected sensitive dataset. We study the relationship between fairness, privacy and utility in Federated Learning.

In Federated Learning (FL), fairness and privacy issues are always hot topics. Studying the relationship between fairness, privacy, and utility in FL is crucial for achieving a secure and reliable FL. This paper mainly explores local fair classifiers with private demographic data in FL and also examines the relationship between fairness, privacy, and utility. Specifically, the paper proposes a differential private fair FL algorithm (FedPF). Here, the FL optimization problem is transformed into a zero-sum game based on Lagrange multipliers. To solve the Nash equilibrium of this game, we uses \textit{Q-Learner} and \textit{$\lambda$-Player} to find the optimal local classifier. Theoretically, we derive the tradeoff between privacy, fairness, and utility in FL and determine the convergence and robustness boundaries satisfied of our algorithm. Especially, we find the privacy and fairness are roughly inversely proportional and utility of the global model is determined by privacy, fairness and the internal parameters, such as the mean error $\widehat{\operatorname{err}}(Y)$ downward with the increase of the privacy budget $\epsilon_p$, but due to the constraints of fairness $\epsilon_f$, the mean error $\widehat{\operatorname{err}}(Y)$ may increase dynamically. 

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
# code
python main.py
```

## Results
We consider three scenerios, including fairness metrics, privacy metrics, fairness and privacy metrics.

The templete is in the file `system\federated_exponential_mechanism_adult.ipynb` and `system\federated_exponential_mechanism_bank.ipynb`

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
