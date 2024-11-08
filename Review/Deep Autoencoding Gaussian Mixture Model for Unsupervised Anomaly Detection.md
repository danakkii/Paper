# Abstract

<aside>
💡 1. 분리된 모델 학습에서 일관적이지 않은 최적화 목표
2. 저차원 공간에서 중요한 정보를 제대로 보존 못함
compression network : 저차원 공간으로서의 차원 축소 & reconstruction error 생성
→ deep autoencoder (non-linearity)
estimation network : compressed data의 확률분포 추정
→ gaussian mixture model 사용
autoencoder와 mixture model을 동시에 같이 학습할 수 있는 end-to-end 방식 제
em대신 joint optimisation 활용

</aside>

# Unsupervised Anomaly Detection

[]()

one-class supervised anomaly detection 방식은 정상 sample이 필요하다. 어떤 것이 정상 sample인지 알기 위해서는 반드시 정상 sample에 대한 label을 확보하는 과정이 필요하다. 대부분의 데이터가 정상 sample이라는 가정을 하여 label 취득없이 학습을 시키는 unsupervised anomaly detection 방법론 연구

가장 단순하게는 주어진 데이터에 대해 Principal Component Analysis(PCA, 주성분 분석)를 이용하여 차원을 축소하고 복원을 하는 과정을 통해 비정상 sample을 검출할 수 있습니다. , Neural Network 기반으로는 대표적으로 Autoencoder 기반의 방법론이 주로 사용되고 있습니다. Autoencoder는 입력을 code 혹은 latent variable로 압축하는 Encoding과, 이를 다시 원본과 가깝게 복원해내는 Decoding 과정으로 진행이 되며 이를 통해 데이터의 중요한 정보들만 압축적으로 배울 수 있다는 점에서 데이터의 주성분을 배울 수 있는 PCA와 유사한 동작을 한다고 볼 수 있습니다.

Autoencoder를 이용하면 데이터에 대한 labeling을 하지 않아도 데이터의 주성분이 되는 정상 영역의 특징들을 배울 수 있습니다. 이때, 학습된 autoencoder에 정상 sample을 넣어주면 위의 그림과 같이 잘 복원을 하므로 input과 output의 차이가 거의 발생하지 않는 반면, 비정상적인 sample을 넣으면 autoencoder는 정상 sample처럼 복원하기 때문에 input과 output의 차이를 구하는 과정에서 차이가 도드라지게 발생하므로 비정상 sample을 검출할 수 있습니다.

다만 Autoencoder의 code size (= latent variable의 dimension) 같은 hyper-parameter에 따라 전반적인 복원 성능이 좌우되기 때문에 양/불 판정 정확도가 Supervised Anomaly Detection에 비해 다소 불안정하다는 단점이 존재합니다. 또한 autoencoder에 넣어주는 input과 output의 차이를 어떻게 정의할 것인지(= 어떤 방식으로 difference map을 계산할지) 어느 loss function을 사용해 autoencoder를 학습시킬지 등 여러 가지 요인에 따라 성능이 크게 달라질 수 있습니다. 이렇듯 성능에 영향을 주는 요인이 많다는 약점이 존재하지만 별도의 Labeling 과정 없이 어느정도 성능을 낼 수 있다는 점에서 장단이 뚜렷한 방법론이라 할 수 있습니다.

하지만 Autoencoder를 이용하여 Unsupervised Anomaly Detection을 적용하여 Defect(결함)을 Segment 하는 대표적인 논문들에서는 Unsupervised 데이터 셋이 존재하지 않아서 실험의 편의를 위해 학습에 정상 sample들만 사용하는 Semi-Supervised Learning 방식을 이용하였으나, Autoencoder를 이용한 방법론은 Unsupervised Learning 방식이며 Unsupervised 데이터 셋에도 적용할 수 있습니다. Autoencoder 기반 Unsupervised Anomaly Detection을 다룬 논문들은 다음과 같습니다.

- [**Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders**](https://arxiv.org/pdf/1807.02011.pdf)
- [**Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images**](https://arxiv.org/pdf/1804.04488.pdf)
- [**MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection**](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)

Autoencoder 기반의 anomaly detection 방법론에 대한 설명은 [**마키나락스 김기현님 블로그 글**](https://kh-kim.github.io/blog/2019/12/15/Autoencoder-based-anomaly-detection.html) 에 잘 정리가 되어있어 따로 다루진 않을 예정입니다.

- 장점: Labeling 과정이 필요하지 않다.
- 단점: 양/불 판정 정확도가 높지 않고 hyper parameter에 매우 민감하다.

# joint optimization

1개의 model에서 n개의 output을 출력한다면 이에 맞춰 n개의 서로 다른 loss를 뽑을 수 있다.

여러 개의 loss들을 하나의 값으로 더해서 최종 loss로 사용하는 훈련 방식이다.

각 채널이 서로 다른 역할을 수행하려면 이에 맞게 loss function을 적용해 줘야 한다.

total loss = loss1 + loss2 + loss3 + loss4 + loss5

https://ballentain.tistory.com/30

The joint optimization, which well balances autoencoding reconstruction, density estimation of latent representation, and
regularization, helps the autoencoder escape from less attractive local optima and
further reduce reconstruction errors, avoiding the need of pre-training.

[]()

# alternate training

loss 개수에 맞게 optimizer를 생성해 각 optimizer를 번갈아 가면서 학습하는 방식

[]()

# human supervision

사람이 부여한 정답

Although fruitful progress has been made in the last several years, conducting robust anomaly
detection on multi- or high-dimensional data without human supervision remains a challenging
task.

https://davinci-ai.tistory.com/10

# introduction

To address this issue caused by the
curse of dimensionality, two-step approaches are widely adopted (Candes et al. (2011)), in which `
dimensionality reduction is first conducted, and then density estimation is performed in the latent
low-dimensional space

1. two-step approaches
    1. dimensionality reduction
    2. density estimation

## suboptimal performance

차선 

뭐지..

However, these approaches could easily lead to suboptimal performance,
because dimensionality reduction in the first step is unaware of the subsequent density estimation
task, and the key information for anomaly detection could be removed in the first place.

(2) anomalies are harder to
reconstruct, compared with normal samples. Unlike existing methods that only involve one of the
aspects (Zimek et al. (2012); Zhai et al. (2016)) with sub-optimal performance, DAGMM utilizes a
sub-network called compression network to perform dimensionality reduction by an autoencoder,
which prepares a low-dimensional representation for an input sample by concatenating reduced
low-dimensional features from encoding and the reconstruction error from decoding.

First, DAGMM preserves the key information of an input sample in a low-dimensional space that
includes features from both the reduced dimensions discovered by dimensionality reduction and
the induced reconstruction error. From the example shown in Figure 1, we can see that anomalies
differ from normal samples in two aspects: (1) anomalies can be significantly deviated in the reduced
dimensions where their features are correlated in a different way; and (2) anomalies are harder to
reconstruct, compared with normal samples. Unlike existing methods that only involve one of the
aspects (Zimek et al. (2012); Zhai et al. (2016)) with sub-optimal performance, DAGMM utilizes a
sub-network called compression network to perform dimensionality reduction by an autoencoder,
which prepares a low-dimensional representation for an input sample by concatenating reduced
low-dimensional features from encoding and the reconstruction error from decoding. 

Second, DAGMM leverages a Gaussian Mixture Model (GMM) over the learned low-dimensional
space to deal with density estimation tasks for input data with complex structures, which are yet
rather difficult for simple models used in existing works (Zhai et al. (2016)). While GMM has strong
capability, it also introduces new challenges in model learning. As GMM is usually learned by
alternating algorithms such as Expectation-Maximization (EM) (Huber (2011)), it is hard to perform
joint optimization of dimensionality reduction and density estimation favoring GMM learning, which
is often degenerated into a conventional two-step approach. To address this training challenge,
DAGMM utilizes a sub-network called estimation network that takes the low-dimensional input from
the compression network and outputs mixture membership prediction for each sample. With the
predicted sample membership, we can directly estimate the parameters of GMM, facilitating the
evaluation of the energy/likelihood of input samples. By simultaneously minimizing reconstruction
error from compression network and sample energy from estimation network, we can jointly train a
dimensionality reduction component that directly helps the targeted density estimation task.
Finally, DAGMM is friendly to end-to-end training. Usually, it is hard to learn deep autoencoders
by end-to-end training, as they can be easily stuck in less attractive local optima,

## Expectation-Maximization

- alternating algorithms

[https://modern-manual.tistory.com/entry/EM-알고리즘-예시로-쉽게-이해하기-Expectation-maximization-EM-algorithm](https://modern-manual.tistory.com/entry/EM-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%98%88%EC%8B%9C%EB%A1%9C-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-Expectation-maximization-EM-algorithm)

모수 : 모집단의 수치적 요약값, 모평균이나 모표준편차 같은 모집단에 대한 통계 

정답 label없이 전혀 알지 못하는 hidden variable에 대한 값을 풀 수 있다는 특

## memebrship

To address this training challenge,
DAGMM utilizes a sub-network called estimation network that takes the low-dimensional input from
the compression network and outputs mixture membership prediction for each sample. With the
predicted sample membership, we can directly estimate the parameters of GMM, facilitating the
evaluation of the energy/likelihood of input samples. 

## regularization

- W(weight)가 너무 큰 값들을 가지지 않도록 하는 것이다.
- W가 너무 큰 값을 가지게 되면 과하게 구불구불한 형태의 함수가 만들어지는데, Regularization은 이런 모델의 복잡도를 낮추기 위한 방법이다.
- Regularization은 단순하게 cost function을 작아지는 쪽으로 학습하면 특정 가중치 값들이 커지면서 결과를 나쁘게 만들기 때문에 cost function을 바꾼다.

Our empirical study demonstrates
that, DAGMM is well-learned by the end-to-end training, as the regularization introduced by the
estimation network greatly helps the autoencoder in the compression network escape from less
attractive local optima.

## local optima

Our empirical study demonstrates
that, DAGMM is well-learned by the end-to-end training, as the regularization introduced by the
estimation network greatly helps the autoencoder in the compression network escape from less
attractive local optima.

https://simonezz.tistory.com/87

모델은 트레이닝 과정에서 loss function을 최적화시키면서 훈련된다.

local optima에 빠지면 global optima를 찾아가기 힘들

## end-to-end train

입력부터 출력까지 파이프라인 네트워크 없이 신경망으로 한 번에 처리

- raw data → neural network → prediction

In addition, the end-to-end trained DAGMM significantly outperforms all the baseline methods that rely
on pre-trained autoencoders

https://velog.io/@jeewoo1025/What-is-end-to-end-deep-learning

[]()

모델의 모든 매개변수가 하나의 손실함수에 대해 동시에 훈련되는 경로가 가능한 네트워크로써 역전파 알고리즘 (Backpropagation Algorithm) 과 함께 최적화 될 수 있다는 의미이다.

예를들어 인코더(언어의 입력을 벡터로)와 디코더(벡터의 입력을 언어로)에서 모두가 동시에 학습되는 기계 번역 문제에서 효과적으로 적용 될 수 있다.

즉, 신경망은 한쪽 끝에서 입력을 받아들이고 다른 쪽 끝에서 출력을 생성하는데, 입력 및 출력을 직접 고려하여 네트워크 가중치를 최적화 하는 학습을 종단 간 학습(End-to-end Learning) 이라고 한다.

## explicit linear projections

implicit non-linear projections 

Conventional methods in this category
include Principal Component Analysis (PCA) (Jolliffe (1986)) with explicit linear projections, kernel
PCA with implicit non-linear projections induced by specific kernels (Gunter et al.), and Robust PCA
(RPCA) (Huber (2011); Candes et al. (2011)) that makes PCA less sensitive to noise by enforcing `
sparse structures.

# reconstruction error

However, the performance of reconstruction based methods is limited by the fact that
they only conduct anomaly analysis from a single aspect, that is, reconstruction error.

Although the
compression on anomalous samples could be different from the compression on normal samples
and some of them do demonstrate unusually high reconstruction errors, a significant amount of
anomalous samples could also lurk with a normal level of error, which usually happens when the
underlying dimensionality reduction methods have high model complexity or the samples of interest
are noisy with complex structures. 

### anomaly detection with AE

1. 입력 샘플을 인코더를 통해 저차원으로 압축
2. 압축된 샘플을 디코더를 통과시켜 다시 원래 차원으로 복원
3. 입력 샘플과 복원 샘플의 복원 오차(reconstruction error)를 구함
4. 복원 오차는 이상 점수(anomaly score)가 되어 threshold와 비교를 통해 이상 여부를 결정
    1. threshold 보다 클 경우 이상으로 간주
    2. threshold 보다 작은 경우 정상으로 간주
    
    https://kh-kim.github.io/blog/2019/12/15/Autoencoder-based-anomaly-detection.html
    

Because of the
curse of dimensionality, it is difficult to directly apply such methods to multi- or high- dimensional
data. Traditional techniques adopt a two-step approach (Chandola et al. (2009)), where dimensionality reduction is conducted first, then clustering analysis is performed, and the two steps are separately learned.

1. dimensionality reduction
2. clustering analysis

To address this issue, recent
works propose deep autoencoder based methods in order to jointly learn dimensionality reduction
and clustering components → deep autoencoder를 통해 차원 축소와 군집화를 진행

However, the performance of the state-of-the-art methods is limited by
over-simplified clustering models that are unable to handle clustering or density estimation tasks for data of complex structures, or the pre-trained dimensionality reduction component (i.e., autoencoder) has little potential to accommodate further adjustment by the subsequent fine-tuning for anomaly detection. → 과하게 단순화된 군집 모델은 군집화가 힘들고 차원 측정이 힘들고 사전학습된 차원 축소가 힘들다 

DAGMM explicitly addresses these issues by a sub-network called estimation network
that evaluates sample density in the low-dimensional space produced by its compression network. → DAGMM는 해결하기 위해서 estimation network로 불리는 sub-network로 해결한다. 

By predicting sample mixture membership, we are able to estimate the parameters of GMM without
EM-like alternating procedures. Moreover, DAGMM is friendly to end-to-end training so that we can unleash the full potential of adjusting dimensionality reduction components and jointly improve the quality of clustering analysis/density estimation.

→ sample mixture membership을 예측함으로써 alternating procedures없이 gmm 파라미터를 측정할 수 있다. 

DAGMM 은 end-to-end training 친화적이라서 차원 축소와 군집화를 더 잘 할 수 있다.

# overview

Deep Autoencoding Gaussian Mixture Model (DAGMM) consists of two major components: a compression network and an estimation network.

1. compression network
2. estimation network

the compression network performs dimensionality reduction for input samples by a deep autoencoder, prepares their low-dimensional representations from both the reduced space and the reconstruction error features, and feeds the representations to the subsequent estimation network; (2) the estimation network takes the feed, and predicts their likelihood/energy in the framework of Gaussian Mixture Model (GMM).

## likelihood/energy

the estimation network takes the feed, and predicts their likelihood/energy in the framework of Gaussian Mixture Model (GMM).

With the predicted sample membership, we can directly estimate the parameters of GMM, facilitating the evaluation of the energy/likelihood of input samples.

https://jaejunyoo.blogspot.com/search/label/kr

https://velog.io/@pabiya/ENERGY-BASED-GENERATIVE-ADVERSARIAL-NETWORKS

https://velog.io/@jsshin2022/Paper-Review-Deep-Autoencoding-Gaussian-Mixture-Model-for-Unsupervised-Anomaly-Detection

https://iamseungjun.tistory.com/22

reconstruction error를 energy로 간주한다. 

→ loss function이랑 비슷한 건가?

## estimation network

- sub-network

the estimation network estimates the parameters of GMM and evaluates the likelihood/energy for samples without alternating procedures such as EM (Zimek et al. (2012)). The estimation network achieves this by utilizing a multi-layer neural network to predict the mixture membership for each sample.
