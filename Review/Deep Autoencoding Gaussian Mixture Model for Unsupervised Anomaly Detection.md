# Abstract - 5min

- problem
    
    It has been observed that sometimes the autoencoder “generalizes” so well that it can also reconstruct anomalies
    well, leading to the miss detection of anomalies.
    
- motivated
    
    To mitigate this drawback for autoencoder based anomaly detector, we propose to augment the autoencoder with a memory module and develop an improved autoencoder called
    memory-augmented autoencoder, i.e. MemAE.
    
- contributed
    
    The reconstruction will thus tend to be close to a normal sample. Thus the reconstructed errors on anomalies will be strengthened for anomaly detection.
    
- novelty
    
    MemAE is
    free of assumptions on the data type and thus general to be
    applied to different tasks
    

input = encoding from the encoder → query

- training stage

At the training stage, the
memory contents are updated and are encouraged to represent the prototypical elements of the normal data. 

- test stage

At the test stage, the learned memory will be fixed, and the reconstruction is obtained from a few selected memory records of the normal data. The reconstruction will thus tend to be close to a normal sample.

# Introduction

- [ ]  AE “generalize”

Deep autoencoder (AE) [2, 16] is a powerful tool to
model the high-dimensional data in the unsupervised setting. It consists of an encoder to obtain a compressed encoding from the input and a decoder that can reconstruct the
data from the encoding. The encoding essentially acts as an
information bottleneck which forces the network to extract
the typical patterns of high-dimensional data. In the context
of anomaly detection, the AE is usually trained by minimizing the reconstruction error on the normal data and then
uses the reconstruction error as an indicator of anomalies.
It is generally assumed [47, 9, 44] that the reconstruction
error will be lower for the normal input since they are close
to the training data, while the reconstruction error becomes
higher for the abnormal input.

→ powerful tool to model the high-dimensional data(ex.video) in the unsupervised setting.

input으로부터 compressed encoding, decoder는 encoding data를 reconstruct함

인코딩은 bottleneck을 지나야함 bottleneck은 네트워크를 전형적인 고차원데이터를 추출함

AE는 reconstruction error를 최소화하는 쪽으로 훈련함 정상에 가까울 수록 reconstruction error는 낮아지고 비정상에 가까울수록 올라감.

However, this assumption may not always hold, and
sometimes the AE can “generalize” so well that it can also
reconstruct the abnormal inputs well. → AE “generalize” 역시 비정상을 재구성

비정상 training sample없이 비정상이 복원오차가 높게 나오는 이유 : common compositional patterns 정상 훈련 데이터 또는 디코더가 “too strong” 해서이다.

 

### input

To mitigate the drawback of AEs, we propose to augment the deep autoencoder with a memory module and introduce a new model memory-augmented autoencoder, i.e.
MemAE. Given an input, MemAE does not directly feed its
encoding into the decoder but uses it as a query to retrieve
the most relevant items in the memory. Those items are
then aggregated and delivered to the decoder. Specifically,
the above process is realized by using attention based memory addressing. We further propose to use a differentiable
hard shrinkage operator to induce sparsity of the memory
addressing weights, which implicitly encourage the memory items to be close to the query in the feature space

→ input이 주어질 때 바로인코딩에서 디코더로 넘어가지 않고 인풋을 쿼리로 사용해서 retrieve the most relevant items in the memory. → 이 아이템(쿼리?)은 종합되어 디코더에 전달된다. 이 과정에서 memory addressing 기반된 attention이 사용된다. → 이를 통해 미분가능한 hard operator 감소와 memory addressing weight 희소를 유도한다. → 이 것은 memory items을 query in the feature space에 가깝게 한다. 

### training

In the training phase of MemAE, we update the memory
content together with the encoder and decoder. Due to the
sparse addressing strategy, the MemAE model is encouraged to optimally and efficient use the limited number of
memory slots, making the memory to record the prototypical normal patterns in the normal training data to obtain
low average reconstruction error (See Figure 3). In the test
phase, the learned memory content is fixed, and the reconstruction will be obtained by using a small number of the
normal memory items, which are selected as the neighborhoods of the encoding of the input. Because the reconstruction is obtained normal patterns in memory, it tends to be
close to the normal data. Consequently, the reconstruction
error tends to be highlighted if the input is not similar to
normal data, that is, an anomaly. The schematic illustration is shown in Figure 1. The proposed MemAE is free of
the assumption on the types of data and thus can be generally applied to solve different tasks. We apply the proposed
MemAE on various public anomaly detection datasets from
different applications. Extensive experiments prove the excellent generalization and high effectiveness of MemAE.

→ 훈련 과정에서 인코더와 디코더의 메모리를 업데이트한다. sparse addressing strategy덕분에 한정된 메모리 슬록을 효율적으로 사용할 수 있다. 정상 데이터 안의 정형적인 정상 패턴에서 낮은 복원오차를 얻을 수 있다.

In the test phase, the learned memory content is fixed, and the reconstruction will be obtained by using a small number of the
normal memory items, which are selected as the neighborhoods of the encoding of the input. Because the reconstruction is obtained normal patterns in memory, it tends to be
close to the normal data.

Consequently, the reconstruction error tends to be highlighted if the input is not similar to
normal data, that is, an anomaly.

proposed to jointly model the encoded features and
the reconstruction error in a deep autoencoder. Although the
reconstruction based methods have achieved fruitful results,
their performances are restricted by the under-designed representation of the latent space.

# 2. Related Work - 20min

## Memory Networks

Memory-augmented networks have attracted increasing interest for solving different problems. 

use external memory to extend the capability of neural networks, in which content-based attention is used for addressing the memory.

The external memory has also been used for
multi-modal data generation [14, 20], for circumventing the
mode collapse issue and preserving detailed data structure.

→ external memory를 사용해서 neural network의 수용성을 확장한다. 이것은 content-based attention is used for addressing the memory.

### external memory

# 3. Memory-augmented Autoencoder

## 3.1. Overview

1. encoder
2. decoder
3. memory module : retrieves the most relevent items in the memory via the attention-based addressig operator(delivered to the decoder for reconstruction.)
