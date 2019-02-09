---
layout: default
title: 작성 가이드
---

TmaxAI 블로그는 [Github Pages](https://pages.github.com)와 정적 페이지 생성기인 [Jekyll](https://jekyllrb-ko.github.io/)을 이용해서 서비스됩니다. Github Pages 에서 Jekyll 엔진을 내장하고 있기 때문에 master 브랜치에 마크다운으로 글을 쓰면 Jekyll 엔진이 사전 정의된 형식으로 웹 페이지를 생성해 줍니다.

> 글 작성 뿐만 아니라 블로그에서 버그를 해결하거나 기능을 추가할 때도 비슷합니다.

### Github 블로그 저장소 받아오기

블로그 역시 Github에 저장소 형태로 업로드되어 있기 때문에 글을 쓰기 위해서는(또는 블로그를 수정하려면) 저장소를 가져와야 합니다.

(모두가 알겠지만) 로컬 작업공간에서 다음 커맨드로 저장소를 받아옵니다:

~~~ shell
> git clone https://github.com/tmaxai/tmaxai.github.io.git
~~~

### 로컬 테스트를 위해 Jekyll 설치

받아온 저장소에서 글을 쓰거나 블로그를 수정했을 때 사이트에 어떻게 보일지 테스트할 필요가 있습니다. Github Pages의 master 브랜치는 실제로 외부에 글이 보여지는 곳이기 때문에 매번 master에 수정사항을 올려서 확인하면 **안됩니다**. Jekyll 엔진을 로컬에 설치하면 수정사항을 내 작업환경 안에서 테스트 할 수 있습니다. Jekyll 설치 가이드를 참고해서 설치하세요:

[Jekyll 설치 가이드](https://jekyllrb-ko.github.io/docs/installation/)

설치가 끝났으면 블로그 저장소에서 다음 명령을 실행합니다:

~~~ shell
### Jekyll 빌드 명령어
> bundler exec jekyll build    # 저장소 정보로 페이지를 생성함
> bundler exec jekyll serve    # 로컬에서 웹 서버를 띄우기
~~~

jekyll serve 명령어를 실행하면 http://127.0.0.1:4000/ 경로로 웹 서버가 실행됩니다. 여기에서 수정 사항을 확인하면서 글을 작성할 수 있습니다.

### 글 작성전 git 설정하기

원격 저장소 master 브랜치에는 직접 push 할 수 없습니다. (실수로 버그가 있는 상태로 푸시했을 때 블로그가 고장나는걸 막기 위해서 막아뒀습니다) 글 작성은 다음 순서로 진행하면 됩니다:

#### git 계정 설정

Git 에서 커밋과 푸시할 때 누가 작성한 것인지 확인하기 위해 본인 계정을 설정해야 합니다(계정 설정은 한번만 하면 됩니다).

깃허브에 등록된 계정 이름과 이메일을 사용해야 원격 저장소에서 정보가 제대로 보입니다. 만약 저장소마다 다른 계정명을 사용해야 하는 경우(깃허브 블로그와 내부망 깃랩을 한 컴퓨터에서 작업하는 경우 등) `--global` 대신 `--local` 옵션을 해당 저장소에서 사용하면 그 저장소만 로컬로 설정한 계정 이름과 이메일로 사용할 수 있습니다.

~~~ shell
### 깃 계정 정보 설정
> git config --global user.name "깃허브 계정 이름"
> git config --global user.email "깃허브 계정 이메일"
~~~

#### 로컬 브랜치 생성

~~~ shell
> git checkout -b "브랜치 이름"  # 브랜치 이름으로 새 브랜치를 만들고 이동
~~~

> 브랜치 이름은 글 제목이나 관련 이슈번호+설명으로 작성하세요. ex) 42-contributing_guide, review_faster_rcnn

### 글 작성하기

양식에 맞춰 글을 작성해야 Jekyll 엔진으로 글을 변환할 수 있습니다.

#### Markdown 파일 생성

글은 `_posts` 폴더 안에 마크다운 으로 작성하면 되고 파일명은 다음 양식을 지켜서 작성하면 됩니다:

**YYYY-MM-DD-글 제목.md**

2. YAML 작성

YAML은 Jekyll 엔진이 페이지를 변환할 때 사용하는 메타정보를 담고 있으며 마크다운 맨 위에 위치합니다:

~~~ yaml
### YAML 작성 가이드
---
layout: post
title: "글 제목"
date: YYYY-MM-DD
categories: [카테고리]
tags: [태그 1, 태그 2, 태그 3]
author: ssunno
---
~~~

카테고리와 태그는 글 분류에도 사용되므로 너무 길지 않고 다른 글과 유사하게 맞추는 것이 좋습니다. 카테고리와 태그는 다음 가이드를 참고해서 작성해주세요:

**Categories:**
* Tmax AI Research: 각 팀에서 연구하는 주제와 직접적인 밀접한 관련이 있는 논문들에 대한 포스팅(ex 1팀: Anomaly detection 관련 논문 포스팅 2팀: 얼굴인식. traffic 관련 논문 포스팅, 3팀: NLP 관련 논문 포스팅)
* Paper Review: AI와 관련된 비교적 최신의 흥미로운 논문들에 대한 포스팅
* Tutorial: 비교적 잘 알려진 연구 주제 or 네트워크에 관한 본인만의 내용 정리가 담긴 포스팅

**Tag:**
* 주제: CNN, RNN, LSTM, Autoencoder, Representation Learning 등의 큰 주제를 태그로 달기
* 세부 주제: Attention, Intent Classification, AI-ET, Calibration과 같은 세부 주제들을 태그로 달기
* Keyword: 포스팅한 글의 제목이나 내용 중에서 핵심 키워드를 정리하여 태그로 달기
* 난이도: 초급, 중급, 고급 중 포스팅하는 연구원이 판단한 난이도 태그로 달기

저자 정보는 블로그 `_data/authors.yml`에 사전 등록해야 포스트에 이름이 나타납니다.

#### 글 작성

글은 마크다운 포맷에 맞춰 자유롭게 작성하면 됩니다. 글에 이미지나 유튜브 링크 등을 넣는 방법은 다음과 같습니다:

![import img guide](/assets/img/contributing/import_img.png)

> 글에 이미지를 넣을 때는 저장소의 해당 위치에 이미지를 넣고, 자료 첨부할 때는 태그 위아래에 빈 칸을 넣어주세요. 슬라이드쉐어는 iframe 임베드 코드 전체를 마크다운에 붙여넣기 하세요.

### 테스트

작성한 글이 잘 만들어졌는지 테스트하려면 Jekyll 엔진으로 빌드하면 됩니다:

~~~ shell
### Jekyll 빌드 명령어
> bundler exec jekyll build    # 저장소 정보로 페이지를 생성함
> bundler exec jekyll serve    # 로컬에서 웹 서버를 띄우기
~~~

> build 명령어는 _config.yml 처럼 블로그 설정 자체가 변경되는 경우에만 수행하면 됩니다.

### 글 발행요청

글이 다 작성됐으면 커밋 메시지를 작성해서 원격 저장소로 보내면 됩니다. master 브랜치에는 작업할 수 없기 때문에 글 작성 전에 만든 브랜치를 원격 저장소에 푸시하고 풀 리퀘스트를 요청하면 됩니다.

커밋 메시지를 작성할 때는 (관련 이슈가 있는 경우) 이슈 번호를 포함하고 이번 커밋에서 작업한 내용들을 기록하세요(남들이 볼 수 있도록).

원격 저장소에 푸시하면 Github 저장소에서 Pull Request를 진행할 수 있습니다. 풀 리퀘스트를 작성한 다음 리뷰어를 지정하면 글 검토 후 리뷰어가 master 브랜치에 병합합니다. 그러면 블로그에서 작성한 글을 확인할 수 있습니다.

> master 브랜치에 직접 올리지 마세요 !(admin 권한을 가진 사람)
