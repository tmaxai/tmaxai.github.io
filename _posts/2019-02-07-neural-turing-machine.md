---
layout: post
title:  "Neural Turing Machine"
date:   2019-02-07 
categories: [Research Article]
tags: [Deep Learning, Research Article, Attention] 
author: y-rok
---

이번 포스트에서 소개드릴 논문은 Google Deepmind에서 2014년에 발표한 [Neural Turing Machine](https://arxiv.org/abs/1410.5401)입니다. 아래 
[발표 영상](#발표-영상)
을 보고 [질문 & 답변](#질문--답변) 글을 참고하시기 바랍니다. 

**질문 혹은 토론하고 싶은 사항이 있다면 댓글 부탁드립니다. :)**

## 발표 영상 

### [발표 자료 PDF](https://www.slideshare.net/RokJang/neural-turing-machine-130440568)

{% include youtube.html id="Oatth0OvCss" %} 

## 질문 & 답변
### Read와 Write의 Weight은 다른 Vector 인가?

- Read는 N개의 Head, Write는 M개의 Head를 갖을 수 있음 (N과 M은 Hyper parameter)
- **Head 별로 각각 따로 Weight 존재** / Read & Write 는 각각 Head가 다르므로 Weight도 다름

### Previous Read Vector는 어떻게 Controller의 Input으로 넣어 주는가?

- [Open Source](https://github.com/snowkylin/ntm)에 따르면 Controller의 Input은 External Input과 Previous read vector를 Concat하여 구성됨

### Controller의 External Output은 어떻게 계산 되는가?

- [Open Source](https://github.com/snowkylin/ntm)에 따르면 Controller의 Output을 목적에 따라 다르게 활용하는 2개의 Dense Layer가 존재
    - O2P Layer - Controller의 Ouput을 Parameter로 변환
    - O2O Layer - Controller의 Output을 external output으로 변환

### Head의 의미는?

- Turing Machine에서 Head란 Memory에서 특정 위치에 정보를 Write 혹은 Read 하는 장치
- Neural Turing Machine에서도 유사하게 사용
    - Write 혹은 Read 시 Attention을 위해 각 element에 가중치를 부여하는 Weight Vector (즉, Memory에서 접근할 위치를 결정)

### Multiple Head는 어떻게 동작하는가?

- Controller에서 Head 별로 Parameter가 Output으로 추출되고 이를 활용하여 각 Head는 Read와 Write를 진행
- Read Operation
   - Multiple Read heads는 각 Head의 Read Vector를 Concat하여 최종적으로 1개의 Read Vector 구성
- Write Operation
   - Multiple Write heads는 head 수 만큼 순차적으로 Memory에 Write operation을 수행

### Multiple Head를 사용하는 이유?

- **복잡한 알고리즘을 수행하기 위해서 1번의 time step에서 여러번의 Read, Write Operation이 필요할 수 있음**
- 예를 들어, Copy Task같은 단순한 알고리즘을 수행하기 위해서는 각 time step에서 Read 혹은 Write 1개의 Operation만 필요하지만 Sorting Task를 수행하기 위해서는 Write 했던 것을 메모리에서 다른 위치로 옮기고 등의 복잡한 Operation 필요
- 따라서, Task에 따라 Head 수를 Hyper-parameter로 결정


## References

- 설명 블로그 (영어)
   - [A Stable Neural-Turing-Machine (NTM) Implementation (Source Code and Pre-Print)](https://www.scss.tcd.ie/joeran.beel/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation-source-code-and-pre-print/)
   - [Neural Turing Machines : an "artificial" working memory ?](https://medium.com/@benjamin_47408/neural-turing-machines-an-artificial-working-memory-cd913420508b)
   - [Neural Turing Machines: a fundamental approach to access memory in deep learning](https://medium.com/@jonathan_hui/neural-turing-machines-a-fundamental-approach-to-access-memory-in-deep-learning-b823a31fe91d)
- 설명 블로그 (한글)
   - [Neural Turing Machine](https://norman3.github.io/papers/docs/neural_turing_machine.html)
- Open Source
   - [snowkylin/ntm](https://github.com/snowkylin/ntm)