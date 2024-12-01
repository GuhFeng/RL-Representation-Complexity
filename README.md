# Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity

This is the repository for the paper [Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity](https://arxiv.org/abs/2312.17248).

We empirically examine the representation complexity of model, optimal policy, and optimal value functions in various simulated MuJoCo environments.

<p align="center">
    <img src=" " width="600">
        <br>
    <em>Figure 1: Main results on various MuJoCo environments.</em>
</p>


## Installation
Set up the environment.

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

## Acknowledgement

This repository was built upon [TD3](https://github.com/sfujim/TD3).

### Citation
If you find the content of this repo useful, please consider cite it as follows:

```
@article{feng2023rethinking,
  title={Rethinking Model-based, Policy-based, and Value-based Reinforcement Learning via the Lens of Representation Complexity},
  author={Feng, Guhao and Zhong, Han},
  journal={arXiv preprint arXiv:2312.17248},
  year={2023}
}
```
