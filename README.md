# Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity

The official code of the paper [Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexit](https://arxiv.org/abs/2312.17248).

## Installation

```
conda env create -f env.yaml  
```

## Usage

Train the oracle policy model by TD3.

```
bash ./train_TD3.sh 
```

Generate the dataset by the policies.

```
bash ./rollout.sh
```

Compute the representation error.

```
bash ./repr_error.sh
```

## Acknowledge

This repository was built upon [TD3](https://github.com/sfujim/TD3).

### Bibtex

```
@inproceedings{feng2023rethinking,
  title={Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity}, 
  author={Guhao Feng and Han Zhong},
  booktitle={Neural Information Processing Systems},
  year={2024}
}
```
