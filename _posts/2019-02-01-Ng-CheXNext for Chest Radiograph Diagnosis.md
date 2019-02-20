---
layout: post
title: "Deep Learning for Chest Radiograph Diagnosis: A Retrospective Comparison of the CheXNeXt Algorithm to Practicing Radiologists"
date: 2019-02-01
categories: [Paper Review]
tags: [CNN, Medical AI, Classification, 초급]
author: hyunsuky
---

이번 포스팅은 다음 논문들의 내용을 이용하여 정리하였습니다.
- [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/pdf/1711.05225.pdf)
- [Deep Learning for Chest Radiograph Diagnosis: A Retrospective Comparison of the CheXNeXt Algorithm to Practicing Radiologists](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0204155&type=printable)

### Introduction
* Chest X-Ray (CXR)는 간단한 절차로도 흉부의 여러 질환을 검진하고 진단할 수 있기 때문에 전 세계적으로 널리 사용되고 있습니다. 매년 약 20억 개의 CXR가 얻어지고 있고, CXR를 판독하는 데 많은 의료 자원이 들어가고 있습니다. 현직 영상의학과 의사들과 비슷한 성능을 낼 수 있는 자동화된 판독 시스템이 개발되면 의료 자원이 부족한 개발도상국에도 사용될 수 있고, 선진국에서도 영상의학과 의사들을 보조하는데 사용될 수 있을 것입니다. 
* 본 논문에서는 최근 미국 보건복지부 (The National Institute of Health)에서 공개한 *ChestX-ray 14 dataset*을 이용하여 deep learning model인 CheXNeXt를 학습시키고 CXR에서 가장 흔하게 발견되는 *14개 종류*의 병변 (결핵, 흉수, 결절, 종괴 등)에 대해 영상의학과 의사들과 성능 비교를 하였습니다. 
* 참고로 ChestX-ray 14 dataset은 30805명의 환자들에 대해 찍은 112120개의 데이터셋으로 이루어져 있고, 각 14개의 병에 대한 label이 달려 있는 public dataset입니다.

### Methods
* Network structure: CheXNeXt는 **121-layer DenseNet** architecture이고, input image로 512 pixel by 512 pixel로 resize 및 normalize된 CXR를 받아서 각 14개 종류의 병변이 있을 확률을 output으로 냅니다. 이러한 output을 내기 위해서 기존 DenseNet의 마지막 fully-connected layer를 변형하여 14-dim output을 내는 fully-connected layer로 변형하였습니다.  
* Network training: ChestX-ray 14 dataset은 잘못된 label이 있을 수 있으므로, 이를 보정하기 위해 **두단계로 학습이 진행**되었습니다. 
    * 첫번째 단계에서는 training set에서 여러 개의 network가 학습되었고, 높은 성능을 보이는 network subset으로 *ensemble*을 구성하였습니다. 각 subnetwork의 output prediction을 평균하여 전체 ensemble의 output probability를 얻었고, 전체 14개 병변에 대해 가장 높은 F1 score를 줄 수 있는 threshold 값을 정하여 probability value를 binary value로 변환하였습니다. 이 output은 두번째 단계 학습에서의 label로 이용되었습니다.
    * 두번째 단계에서는 relabeled data에서 10개의 network를 다시 학습시켜 ensemble을 구성하였습니다. 첫번째 단계에서처럼 각 network의 output prediction을 평균하여 output probability를 얻었습니다. Regularization을 위해 input 이미지를 50% 확률로 lateral inversion을 하였습니다.
    * Loss function: loss function은 **weighted binary cross entropy loss**를 사용하였습니다. P: positive case, N: negative case라 할 때 loss function의 정의는 다음과 같습니다.
    $$L(X,y) = -w_+·ylogp(Y=1|X)-w_-·(1-y)logp(Y=0|X)$$ 
    $$w_+= N/(P+N), w_-= P/(P+N)$$
* Class activation mapping (CAM): CVPR 2016에 발표된 논문 "Learning Deep Features for Discriminative Localization"에 소개된 대로 CAM을 구현하여 heat map을 얻었습니다.
* 영상의학과 의사와 성능 비교: ChestX-ray 14 dataset에서 각 병에 대해 적어도 50개 이상의 이미지를 포함하도록 420개의 영상 이미지가 test dataset으로 선택되었습니다. Test dataset에 대해 3명의 흉부 specialist 영상의학과 의사가 14개의 병변에 대해 각각 label을 달았고, majority vote으로 ground truth label을 정하였습니다. 

### Results
* CheXNeXt 네트워크는 10개 종류의 병변에 대해 영상의학과 의사와 대등한 성능을 보였고, 3개의 종류의 병변에 대해서는 영상의학과 의사에 비해 낮은 성능을 보였으며, 1개의 병변에 대해서는 높은 성능을 보였습니다. 영상의학과 의사에 비해 낮은 성능을 보인 병변의 종류로 *심비대, 폐기종, 식도열공탈장*이 있고, 영상의학과 의사에 비해 높은 성능을 보인 병변으로는 *무기폐*가 있습니다.
* 420개의 이미지를 판독하는데 영상의학과 의사는 6시간이 걸렸지만, CheXNeXt는 1.5분 밖에 소요되지 않았습니다. 

![AUC](/assets/img/2019-02-01-Ng-CheXNeXt/AUC.PNG)

* Class activation mapping 결과를 보면 CheXNeXt는 정확하게 영상의학적 병변을 classify할 뿐만 아니라 localize도 할 수 있는 것을 알 수 있습니다. 아래 그림은 각각 폐결절과 폐결핵 병변을 정확하게 localize한 결과입니다. 

![Heat-map](/assets/img/2019-02-01-Ng-CheXNeXt/heat_map.PNG)

### Conclusion
* 본 연구 결과는 Deep Learning을 이용하여 영상의학과 의사 수준으로 CXR 판독이 가능한 것을 시사합니다. CheXNeXt는 영상의학과 의사가 CXR에서 놓치기 쉬운 결핵이나 폐암의 병변을 검출하는데 사용할 수 있을 것입니다. 
* CheXNeXt의 단점으로는 폐기종처럼 폐병변이 특정 폐야에 국한되어 있지 않고, global하게 있는 경우 성능이 크게 떨어질 수 있고, 학습 데이터가 불충분하면 성능 저하가 있다는 점입니다. 또한 실제 임상 상황에서는 환자 정보 (나이, 혈액 검사 수치 등)가 쓰이지만, ChexNeXt는 이러한 환자의 다양한 정보를 활용하지 않았다는 것도 큰 단점입니다. 




