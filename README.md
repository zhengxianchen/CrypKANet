# CrypticKANet

Accurately predicting protein cryptic binding sites via Equivariant Graph Neural Networks and Gated Attention Mechanisms

## Usage

Here we provide instructions for two use cases: (1) Retraining our model on our or your data. (2) Testing data on trained models.

## Train CrypticKANet

### Key Environment

```
Pytorch: 2.7.0
```

After preparing your data or opting to use our provided dataset, execute the following command to start the model training.

```bash
python train.py
```

## Test

You can use the provided model file 'src/ckpt/best_model.pth' or your retrained model to perform testing.

After preparing the test set, please run 

```bash
python inference.py
```

 to start testing.

## Acknowledge

Our code is built upon [Gated-GPS](https://github.com/gxx27/Gated-GPS), we thank the authors for their open-sourced code.