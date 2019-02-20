---
layout: post
title: "Learning Transferable Architectures for Scalable Image Recognition" 
date: 2019-02-20
categories: [Tmax AI Research]
tags: [CNN, Transferable architecture, ImageNet, 중급]
author: hyunsuky
---
이번 포스팅은 다음 논문들의 내용을 이용하여 정리하였습니다.
- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)

## 요약
* 소개해드릴 논문은 CVPR 2018년에 발표된 논문입니다. 
* 본 논문에서는 작은 dataset의 search space에서 성능이 좋은 model architecture를 찾은 뒤, 해당 architecture를 더 큰 dataset에서 transfer 시킨 결과를 소개하고 있습니다. 
* 저자들이 제안하는 **"NASNet architecture"**는 CIFAR-10 dataset에서 탐색으로 찾은 cell을 stacking하여, ImageNet dataset에도 적용한 convolutional architecture 입니다. NASNet은 ImageNet datset에서 state-of-the-art accuracy를 보이면서, FLOPS는 현저하게 낮은 구조입니다. 
* Image classification task를 통해 학습된 image feature들을 object detection에도 이용하여, COCO dataset에서 state-of-the-art 보다도 4.0% 높은 mAP를 얻었습니다. 

## Introduction
* 최근에 [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1610.02391.pdf)에서 제안된 **Neural Architecture Search (NAS) framework**는 reinforcement learning search method를 이용하여 architecture configuration을 optimize 시키는 방법입니다. 하지만 NAS를 큰 dataset에 적용하면 computational cost가 매우 크다는 단점이 있습니다. 
* 본 논문은 이러한 단점을 극복하기 위해서 ImageNet보다 훨씬 작은 dataset인 CIFAR-10 dataset에서 architecture 탐색을 시행하였고, search space의 network를 동일한 구조의 convolutional structure(**"cell"**)로 한정하였습니다. 

## Method
* 논문에서 사용하는 architecture 탐색 방법은 **Neural Architecture Search (NAS) framework**입니다. NAS에서는 **controller RNN**이 다양한 architecture를 갖는 child network를 sampling하고, 각 child network가 validation set에서 보이는 accuracy를 이용하여 controller weight가 policy gradient로 update됩니다. 

![NAS framework](/assets/img/2019-02-20-Zoph-transferable-architecture/NAS.png){:height="50%" width="50%"}

* Controller RNN을 이용해서 CNN architecture에서 흔히 발견되는 motif들 *(convolutional filter banks, nonlinearities, connections, etc.)*로 표현되는 **generic convolutional cell**을 예측하고, 그러한 cell들을 stack시키면 어떠한 spatial dimension과 filter depth도 처리할 수 있는 scalable architecture를 만들 수 있게 됩니다.
* 이러한 scalable architecture를 만들기 위해서는 두가지 종류의 convolutional cell이 필요하고, 두 종류의 cell은 모두 다른 architecture를 갖게 됩니다. 
1. **Normal cell**: normal cell은 같은 dimension의 feature map을 return하게 됩니다. 
2. **Reduction cell**: reduction cell은 height과 width의 크기가 반으로 줄어 든 feature map을 return하게 됩니다. 

![Scalable architecture](/assets/img/2019-02-20-Zoph-transferable-architecture/scalable-architecture.png){:height="50%" width="50%"}

* 참고로 CIFAR-10에 비해 ImageNet은 reduction cell의 갯수가 많은데, 그 이유는 image size가 각각 299x299와 32x32이기 때문입니다. 
* Controller RNN은 convolutional cell의 structure를 recursive하게 예측하게 된다. 
    * Step 1. 전 lower layer의 output을 hidden state으로 선택한다.
    * Step 2. Step 1과 같은 과정으로 다른 hidden state을 선택한다. 
    * Step 3. Step 1에 적용할 operation을 선택한다.
    * Step 4. Step 2에 적용할 operation을 선택한다.
    * Step 5. Step 3과 Step 4를 합칠 수 있는 method를 선택한다. 


![Controller prediction process](/assets/img/2019-02-20-Zoph-transferable-architecture/prediction-process.png)




## Results

