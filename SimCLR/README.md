# SimCLR
A Simple Framework for Contrastive Learning of Visual Representations


### Abstract
- SimCLR은 unsupervised learning algorithm으로 이미지 데이터에 label이 없는 상황에서 visual representation을 추출하여 downstream task를 해결하고자 함
- Data augmentation을 통해 얻은 positive/negative sample에 대해 contrastive learning을 적용(positive pair끼리는 같게, negative pair 끼리는 다르게)  같은 데이터에 여러 방식으로 변형하여 얻은 잠재 벡터가 서로 일치하도록 학습

### The Contrastive Learning Framework
![simclr_1](https://github.com/danakkii/Paper/assets/117612063/26df35ab-6e0e-4737-bcd5-ac83448fdd24)

1.  𝑥 →  𝑑𝑎𝑡𝑎 𝑎𝑢𝑔𝑚𝑒𝑛𝑡𝑎𝑡𝑖𝑜𝑛("t’ ~ " τ)→ 𝑥 ̃"i, " 𝑥 ̃"j" (positive pair)
* input = minibatch

2. f(𝑥 ̃"i") → Encoding(ResNet(𝑥 ̃i)) → hi 
* extracts representation vectors(=encoder) 

3. g("hi" )→𝑤^((2))  "σ(" 𝑤^((1)) "hi)"→𝑧𝑖→𝑙𝑜𝑠𝑠 𝑚𝑖𝑛𝑖𝑚𝑢𝑚
*  Maps representations, σ = ReLU non-linearity
*  Vector representation의 유사성은 최대화, contrastive loss function은 최소화  


![simclr_4](https://github.com/danakkii/Paper/assets/117612063/c626131e-7630-4bab-868e-3daea4a3fb6d)

### Algorithm 1 - SimCLR's main learning algorithm
![simclr_2](https://github.com/danakkii/Paper/assets/117612063/4b21e586-b08a-4786-95ef-cb92fea2653f)

### Experiment
-  Defaul setting
Data augmentation : random crop and resize (with random flip), color distortions, and Gaussian blur
Encoder : ResNet-50, 2-layer MLP projection head
           *To project the representation to a 128-dimensional latent space
Loss : NT-Xent, LARS optimizer
* With learning rate of 4.8(=0.3 x batch size/256), weight decay of 10-6
Batch size : 4096 for 100 epochs

### Conclusion
Color + Crop transformer 조합이 가장 성능이 좋음
Contrastive learning의 단점인 Negative pair가 많이 필요하다는 점을 batch size를 크게 하여 해결하였지만 batch size가 커야 성능이 좋다는 한계점 존재


