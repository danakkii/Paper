# Abstract

- novelty
    - prove anomaly detection and segmentation in 1-class and 2-class 5/10/highshot training setups
    - improves these contrastive pre-training baselines and even the supervised pre-training

# 1. Introduction

- problem
    - anomaly detection and segmentation are
    instances of image classification and semantic segmentation problems, respectively, they have unique challenges.
    - First, defects are rare
    - Second, common types of anomalies
    - Third, highly accurate models.
    - Fourth, wide range of domains and tasks
    - trained for a particular object and require re-training
    for different ones.
- contributed
    - due to the rarity of anomalous data, there has been
    a predominant focus on 1-class anomaly detection, which only requires normal images for model training
    - cover different use cases in practice, we establish benchmarks not only in standard 1-class training setup but also 2-class training setups with 5/10/high-shot.
    - Transformations, such as cropping and color jittering, are
    applied globally to the anchor for positives generation.
- novelty
    - The InfoNCE or cosine similarity losses [8,9,23] encourage invariance to these global deformations
    - However, anomaly detection relies on local textual details to spot defects. Thus the subtle and local intra-object (or intra-class) differences
    - As far as we know, improving representations by self-supervision for better downstream anomaly detection/segmentation has not been studied before and we explore this angle.
    - Inspired by the spot-the-difference puzzle shown in Fig
    - sensitive to the subtle differences between the two globally alike images, which is similar to anomaly detection.
    - produce the local perturbations on SPD negatives for synthetic spot-the-difference.
    - negatives as anomaly detection should spot defects under slight global changes in lighting and object pose/position
    - slight global changes as shortcuts to differentiate negatives,
    - applying weak global augmentations on the anchor
    - feature similarities between SPD negative pairs
    - anomalous patterns and invariant to slight global variations.
    1. 1-class and 5/10/high-shot 2-class benchmarks to cover different use cases.
    2. To promote the local sensitivity to anomalous patterns, a SPot-the-Difference
    (SPD) training is proposed to regularize self-supervised ImageNet pre-training,
    which benefits their transfer-learning ability
    3. strong self-supervised pre-training baselines, improves them for better anomaly detection and segmentation.

# 2. Related Work

### Unsupervised Anomaly Detection and Segmentation

- novelty
    - proposed to detect low-level texture anomalies [35], such as scratches and cracks,
    - learns a parametric distribution over patches for anomaly detection.
    - ImageNet models are used in these methods either as feature extractors or initialization for fine-tuning.
    - semantic anomaly detection approaches can be
    less effective for texture anomaly detection as their challenges are different.
    - transfer learning tasks
    - patch features to global augmentations
    - not lead to local sensitivity to tiny defects.
    - generalization ability to surface defect detection tasks

# 3. SPot-the-Difference (SPD) Regularization

- increase model invariance to slight global changes by maximizing the feature similarity between an image and its weak global augmentation, while forcing dissimilarity for local perturbations,
- we first present background in contrastive learning

## 3.1 Background on Self-supervised Contrastive Learning

- Many self-supervised learning methods, such as SimCLR [8] and MoCo [23], are based on contrastive learning
- methods maximize the feature similarity between two strongly augmented samples xi and ˆxi while minimizing the similarities between the anchor xi and other images xj ’s in the same batch of size N
- τ is a temperature scaling hyperparameter.
- share semantics with anchor
- maximize their similarity, the features are forced to be invariant about local details and capture the global semantics.
- n enforced by minimizing similarities between anchor

## 3.2 Augmentations for SPD

- images of a batch in standard contrastive training, are used as negatives.
- SmoothBlend is proposed to produce local deformations.
- color jittering is applied to a cut patch.
- all-zero foreground layer u is created with the patch pasted to a random location. An alpha mask α is created where the pixels corresponding to the pasted patch are set to 1 otherwise 0
- Finally, the augmented sample is obtained by ¯x = (1 − α) ⊙ x + α ⊙ u. ⊙ is the element-wise product.

### Global augmentation

- we use weak global augmentation. Adding global variations to
SPD is motivated by the potentially small global variations in realistic manufacturing environment, such as lighting, object positions, etc
- we choose weak random cropping, Gaussian blurring, horizontal
flipping and color jittering. Such weak global augmentations are different from strong transformations used in SimSiam, SimCLR and MoCo which is illustrated by last two columns in Fig.
- smoothed version
- perturbations
- smooth edges

# 4. Visual Anomaly (VisA) Dataset

## 4.2 Evaluation Protocol and Metrics

- Second, we establish 2-class high/low-shot evaluation protocols as proxies for realistic 2-class setups in commercial products
- In high-shot setup, for each object, 60%/40% normal and anomalous images are assigned to train/test set respectively.
- For low-shot benchmark, firstly, 20%/80% normal and anomalous images are grouped to train/test set respectively.
- Then the k-shot (k=5,10) setup randomly samples k images from
both classes in train set for training.
- Note that for both 1-class and 2-class training setups,
test sets have samples from both classes

# 5. Experiments

### Anomaly detection and segmentation algorithms

- To evaluate the transfer learning performances of different pre-training, we adopt the following algorithms for anomaly detection and segmentation.

## 5.1 SPD in high-shot 1-class / 2-class Regimes

- For the 1-class setting, the results of PaDiM with various pre-training options, both anomaly detection and segmentation across almost all pre-training baselines
- Second, the gap between self-supervised pretraining
- This is in contrast to the low-shot regime
- the 2-class high-shot regime

# 6. Conclusions

- we present a spot-the-difference (SPD) training to regularize pretrained models’ local sensitivity to anomalous patterns.
- demonstrate the benefits of SPD for various contrastive self-supervised and supervised pre-training for anomaly detection
and segmentation. Compared to standard supervised pre-training
- constrative learning
![image](https://github.com/user-attachments/assets/52e4808d-437a-4c15-af9b-52df862da87c)
metric(or distance)는 아래의 조건을 만족해야 한다.
1. Pre-defined Metric
- 우리에게 가장 익숙한, 특정 식으로 인해 도출되는 거리
- 예를 들어, 유클리디안 거리 : *f*(*x*,*y*)=(*x*−*y*)*T*(*x*−*y*)
2. Learned Metric
- 주어진 데이터로부터 얻은(학습한) 특성을 거리에 반영
- 예를 들어, 마할라노비스 거리 : *f*(*x*,*y*)=(*x*−*y*)*TM*(*x*−*y*) (*M* : 데이터로부터 추정된 행렬)
- 본 글에서는 **딥러닝**을 활용하는 방법을 주로 다룰 예정.
