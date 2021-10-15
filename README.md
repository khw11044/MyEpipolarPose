[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-learning-of-3d-human-pose/3d-human-pose-estimation-human36m)](https://paperswithcode.com/sota/3d-human-pose-estimation-human36m?p=self-supervised-learning-of-3d-human-pose)

# Self-Supervised Learning of 3D Human Pose using Multi-view Geometry (CVPR2019) [[project page](https://mkocabas.github.io/epipolarpose.html)]

<https://github.com/mkocabas/EpipolarPose> 에서 사용법 참조 


## Overview
- `scripts/`: includes training and validation scripts.
- `lib/`: contains data preparation, model definition, and some utility functions.
- `refiner/`: includes the implementation of _refinement unit_ explained in the paper Section 3.3.
- `experiments/`: contains `*.yaml` configuration files to run experiments.
- `sample_images/`: images from Human3.6M dataset to run demo notebook.


## Quick start
### Installation
git clone하여 코드 다운   
필요환경 설치
3. Install dependencies using `pip`.
   ```
   pip install -r requirements.txt
   ```
   or create a new `conda` env:
   ```
   conda env create -f environment.yml
   ```
4.  [GoogleDrive](https://drive.google.com/open?id=147AlIWRv9QDmp5pGjwG2yMWEG_2-E2ai) 
(150 MB)  다운로드하여 zip파일 압축을 풀고 `${ROOT}` folder에 넣는다
   ```
   unzip data.zip
   rm data.zip
   ```
5. output 폴더와 models 폴더를 만들어준다.
    ```bash
    mkdir output
    mkdir models
    ```
  

   ```
   ${ROOT}
   ├── data/
   ├── experiments/
   ├── lib/
   ├── models/
   ├── output/
   ├── refiner/
   ├── sample_images/
   ├── scripts/
   ├── demo.ipynb
   ├── README.md
   └── requirements.txt


원본 github에 가서 각 모델을 다운받고 이미지등도 다운받는다.

demo.ipynb가 이상하여 실행할수 있게 demo.ipynb와 demo.py를 만들었다.

$$/begin{bmatrix} 2 & 3 \end(bmatrix}$$
