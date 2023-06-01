## AIST4010 Project
Membership Inference Attack

### Usage
Run experiments
```bash
python main.py
```

Arguments:

`--batch_size`:

`--model_lr`: Target model/ shadow model learning rate.

`--attack_lr`: Attack model learning rate.

`--model_epochs`: Target model/ shadow model epoch.

`--attack_epochs`: Attack model epoch.

`--dataset`: Dataset to use. Currently support `MNIST`, `CIFAR10`, `CIFAR100`, `A1-Kaggle`.

`--target_model`: Target model architecture. Currently support `cnn`, `mlp`, `resnet`, `alexnet`.

`--shadow_model`: Shadow model architecture. Currently support `cnn`, `mlp`, `resnet`, `alexnet`.

`--attack_model`: Attack model architecture. Currently support `mlp`.

`--attack_hidden_size`: Attack model hidden size.

`--model_weight_decay`: Target model/ shadow model weight decay.

`--topk`: Only keep the top k confidence scores.

`--data_aug`: Use data augmentation.

`--mixup`: Use mixup.

`--dp`: Use DP-SGD.

`--load_target`: Load target model from file.

`--load_shadow`: Load shadow model from file.

`--save_model`: Save target model and shadow model to file.

`--no_load`: Do not load target model and shadow model from file.

`--shadow_dataset`: Dataset to use for shadow model. Currently support `MNIST`, `CIFAR10`, `CIFAR100`, `A1-Kaggle`.

`--limit_label`: Only keep data with a limited label (e.g. CIFAR100 with only first 10 classes)

- [x] Metric-based Attacks:
  - [x] Confidence-based Attack
  - [x] Entropy-based Attack
  - [x] Loss-based Attack
- [x] MLP Attack
- [x] Target Model
  - [x] CNN
  - [x] MLP
  - [x] Resnet
  - [x] Alexnet
- [x] Defense
  - [x] Confidence Masking
  - [x] Regularization
  - [x] Augmentation
  - [x] Mix-up
  - [x] DP-SGD
- [x] Datasets
  - [x] MNIST
  - [x] CIFAR10
  - [x] CIFAR100
  - [x] A1-Kaggle
- [x] Experiments
  - [x] Attack accuracies on different models (No defense)
    - [x] MNIST
    - [x] CIFAR10
    - [x] CIFAR100
    - [x] A1-Kaggle
  - [x] Transferability of attacks
    - [x] Different Architecture
      - [x] CIFAR10
      - [x] CIFAR100
      - [x] A1-Kaggle
    - [x] Different Dataset
      - [x] CIFAR10
      - [x] CIFAR100
      - [x] A1-Kaggle
  - [x] Defense ability on generalization gap
    - [x] Confidence Masking
    - [x] Regularization
    - [x] Augmentation
    - [x] Mix-up
- [ ] Code Organization