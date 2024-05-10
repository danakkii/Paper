# SimCLR
A Simple Framework for Contrastive Learning of Visual Representations


### Abstract
- SimCLR은 unsupervised learning algorithm으로 이미지 데이터에 label이 없는 상황에서 visual representation을 추출하여 downstream task를 해결하고자 함
- Data augmentation을 통해 얻은 positive/negative sample에 대해 contrastive learning을 적용(positive pair끼리는 같게, negative pair 끼리는 다르게)  같은 데이터에 여러 방식으로 변형하여 얻은 잠재 벡터가 서로 일치하도록 학습

### The Contrastive Learning Framework
1.  𝑥 →  𝑑𝑎𝑡𝑎 𝑎𝑢𝑔𝑚𝑒𝑛𝑡𝑎𝑡𝑖𝑜𝑛("t’ ~ " τ)→ 𝑥 ̃"i, " 𝑥 ̃"j" (positive pair)
* input = minibatch

2. f(𝑥 ̃"i") → Encoding(ResNet(𝑥 ̃i)) → hi 
* extracts representation vectors(=encoder) 

3. g("hi" )→𝑤^((2))  "σ(" 𝑤^((1)) "hi)"→𝑧𝑖→𝑙𝑜𝑠𝑠 𝑚𝑖𝑛𝑖𝑚𝑢𝑚
*  Maps representations, σ = ReLU non-linearity
*  Vector representation의 유사성은 최대화, contrastive loss function은 최소화
