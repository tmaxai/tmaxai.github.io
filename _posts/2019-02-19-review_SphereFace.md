---
layout: post
title: "[Review] SphereFace: Deep Hypersphere Embedding for Face Recognition"
data: 2019-02-19
categories: [Review]
tags: [Review, Face Recognition]
author: jhlee0427
---
이번 포스팅은 다음 논문들의 내용을 이용하여 정리하였습니다.
* [SphereFace: Deep Hypersphere Embedding for Face Recognition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf)
* [CosFace: Large Margin Cosine Loss for Deep Face Recognition](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1797.pdf)


#### Intro
위 두개의 논문은 기존에 연구되어 오던 closed-set 뿐만 아니라 open-set에서도 좋은 성능을 보이는 metric learning의 연장선이라고 볼 수 있으며, contrastive loss, triplet loss, center loss와 같은 유클리드 공간으로의 매핑이 아닌 반지름 길이 1의 **구**의 공간으로 매핑하여 단순한 거리가 아닌 **각도를 이용해 클래스를 구분하는 방법**을 제시합니다.

빠른 이해를 돕기 위해 softmax loss -> modified softmax loss(Normalized version of Softmax Loss) -> Angular softmax loss -> Large Margin Cosine Loss 순으로 설명을 하고 Euclidean margin loss와 비교를 통해 어떤 점이 개선 되었는지 설명하도록 하겠습니다.

#### Body
##### 개념
![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_2.png)

![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_6.png)

* binary class를 구분한다고 가정했을 때, weights $$W_i$$와 bias $$b_i$$를 가진 softmax loss의 deicision boundary는 
$$(W_1-W2)x+b_1-b_2=0$$로 볼 수 있다. 이를 반지름이 1인 구(Sphere)로 매핑했을 때 $$\Vert W_1\Vert =\Vert W_2\Vert =1, b_1=b_2=0$$이므로 decision boundary는 $$\Vert x\Vert (cos(\theta_1)-cos(\theta_2))=0$$로 표현 될 수 있다. 
* 따라서 만약 우리가 weights와 bias를 
$$(\Vert W_i\Vert =1, b_i=0)$$
으로 normalize한다면 들어오는 인풋이미지 x에 대해서 각각 class의 posterior probabilities는 
$$p_1=\Vert x\Vert cos(\theta_1),p_2=\Vert x\Vert cos(\theta_2)$$
로 표현 할 수 있다. 이에따라 클래스를 구분짓는 
$$\theta_i$$는 $$W_i$$와 $$x$$사이의 각인 
$$\theta_1$$과 $$\theta_2$$에 따라 결정된다.
* posterior probabilities는 $$p_1=\frac{exp(W_1^Tx+b_1)}{exp(W_1^Tx+b_1)+exp(W_2^Tx+b_2)}$$, $$p_2=\frac{exp(W_2^Tx+b_2)}{exp(W_1^Tx+b_1)+exp(W_2^Tx+b_2)}$$로 표현 될 수 있고 일반적인 softmax loss는 $$L=\frac{1}{N}\sum_i-log(\frac{\exp^{f_{y_i}}}{\sum_j \exp^f_j})$$이다.

* 그리고
$$f_j$$
는 $$f_j=W^T_jx=\Vert W_j\Vert  \Vert x\Vert cos\theta_j$$이다. 이를 이용해 수정하면

* modified softmax loss는 
$$L_{modified}=\frac{1}{N}\sum_i-log(\frac{\exp^{\Vert x_i\Vert cos(\theta_{y_i,i})}}{\sum_j\exp^{\Vert x_i\Vert cos(\theta_{j,i})}})$$
이 된다. 이를 Normalized version of Softmax Loss(NSL)이라고도 한다.

* Angular-softamx loss 에선 클래스 1 과 클래스 2의 m margin을 갖는 decision boundary는 
$$\Vert x\Vert (cos(m\theta_1)-cos(\theta_2))=0$$
그리고 
$$\Vert x\Vert (cos(\theta_1)-cos(m\theta_2))=0$$
로 표현 할 수 있다. 

| Softmax | $$\Vert W_1\Vert cos(\theta_1)=\Vert W_2\Vert cos(\theta_2)$$ |
| NSL | $$cos(\theta_1)=cos(\theta_2)$$ |
| A-Softmax | $$C_1:\Vert x\Vert (cos(m\theta_1)-cos(\theta_2))>=0,\\ C_2:\Vert x\Vert (cos(\theta_1)-cos(m\theta_2))<=0$$ |
| LMCL | $$C_1:\Vert x\Vert (cos(\theta_1)-cos(\theta_2))>=m,\\ C_2:\Vert x\Vert (cos(\theta_1)-cos(\theta_2))<=m $$ |


A-Softmax에서는 들어온 인풋이미지의 클래스1이 다른 모든 클래스와의 각도 $$\theta$$보다 m배 축소된 각도내로 제한 시켜서 W를 학습한다.
m은 4가 제일 좋다고 한다.

LMCL에서는 클래스 간의 각도를 상대적인 값인 m배 축소시키는 방법이 아니라 정량적인 값인 +m 으로 설정하고 학습한다.

m은 0에서 0.45사이의 값으로 설정할 수 있다. 0.45를 넘어가면 수렴이 안된다.
m은 0.35가 제일 좋다고 한다.

![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_3.png)

![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_7.png)

###### Weights Vector Normalization & Feature Vector Normalization
피처벡터는 scale 파라미터 s가 magnitude of radius에 의해 조절되는 구에 분포되어 있다.
A-softmax는 L2-norm과 $$cos(\theta)$$를 학습한다.
    - 둘 다 학습한다면 $$cos(\theta)$$가 차이가 너무 작다면 비슷한 L2-distance를 갖는 경우에 같은 클러스터로 분류되는 경우가 생긴다.

LMCL은 전체 셋의 피처벡터를 l2-norm으로 normalize해서 cos(\theta)$$학습에만 집중하게 한다.


###### 특징
m margin은 intra-class의 angular distance라고 볼 수 있고, m이 커지면 구에 매핑되는 한 클래스 이미지들을 더 좁은 지역으로 매핑하게 된다. 모든 metric learning의 목적은 최대화된 intra-class의 angular distance가 최소화된 inter-class의 angular distance보다 작아지게 하는 것이다. 

##### 실험 및 결과
![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_4.png){: width="10" height="10"}

{: width="50%" height="50%"}

- 학습과 추론은 일반적인 방법을 따른다.

![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_5.png){: width="50%" height="50%"}

SphereFace에서 m margin이 증가할수록 인풋 이미지 값들이 구에서 더 큰 마진을 가지며 한곳으로 모여들게 된다. 따라서 클래스 사이에 분명하게 분리되는 곳을 찾을 수 있게 된다. 


![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_8.png)

- 같은 CASIA 훈련 데이터 세트와 같은 64개의 CNN층으로 이루어진 네트워크를 사용해 학습했다고 가정했을 때, 13,233개의 이미지와 5,749개의 클래스를 가진 LFW와 3,425개의 비디오와 1,595의 클래스를 가진 YTF, 그리고 106,863개의 이미지와 530개의 클래스를 가진 MegaFace의 Facescrub 데이터셋에서, A-Softmax와 LMCL이 가장 좋은 성능을 보이고 있다.


![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_9.png)

- 기존에 발표되었던 모델들과 LFW와 YTF에서 verification 정확도를 비교해 보았을 때 SphereFace는 현저히 낮은 데이터 셋(CASIA)으로 학습한 것을 확인 할 수 있다. 성능은 모두 비슷하다.
- CosFace는 0.5M개의 이미지 데이터셋인 CASIA-WebFace와 public 및 private 이미지로 구성된 5M개의 데이터 셋으로 훈련하였다.


#### Conclusion

A-Softmax
- 특징:
    - Angular softmax loss를 처음으로 사용했다.  
- 약점: 
    - $$\theta$$가 작으면 작을 수록 클래스를 구분하기 어렵다.
    - weight vector만 normalize한다.

LMCL
- A-Softmax보다 개선 된 점:
    - $$\theta$$가 작아도 정량적인 m 덕분에 클래스 구분이 보장된다.
    - weight vector뿐만 아니라 feature vector도 normalize한다.
- 단점: 
    - 데이터를 더 많이 사용한다.



#### MegaFace Challenge

- MegaFace challenge는 각자의 데이터로 학습한 수 평가하는 challenge 1과, 주어진 4,700,000개 이미지와 672,000개의 클래스로 이루어진 학습데이터로 학습 후 평가하는 challenge2로 나누어 진다.

- 모델의 identification 및 verification 성능을 평가하기 위해 Megaface, FaceScrub, FGNet 총 3개의 데이터셋을 각각 필요에 따라 사용한다.
    - 갤러리 데이터셋(distractor)
        - MegaFace : 1,000,000개 이상의 이미지 
    - 프로브 데이터셋
        - Facescurb : 100,000개 이상의 이미지  
        - FGNET : 3500개 이상의 이미지 
- small은 500,000개 보다 적은 데이터셋, large는 1,000,000개 이상의 데이터셋으로 훈련한다.
- FaceScrub은 인터넷에서 끌어온 데이터 셋이라 MS-Selec-1M의 있는 celeb의 데이터를 포함한다. 사람 당 200개의 이미지를 가진다.

![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_10.png)

FaceScrub에서 한 클래스를 선택해 pic#n중 pic#1을 뽑아 distractor인 megaface에 더한다. 

![image desc](/assets/img/2019-02-19-review_SphereFace/SphereFace_fig_11.png)

더해진 pic#1와 함께 모든 1mln의 megaface 이미지들의 피처를 뽑아 FaceScrub의 pic#2의 피처와 거리를 비교한다.

현재는 challenge1의 faceScrub 데이터셋 평가에서 정확도 99%의 더 좋은 성능을 보이는 모델들이 많지만 FGNet은 여전히 FaceNet이 정확도 75%수준으로 상위권에 위치한다. Challenge2는 faceScrub, FGNet모두 TencentAIlab(CosFace)이 70% 전후의 정확도를 보이며 1위를 하고 있다.