# Kaggle Pet Adaptation Prediction

Pet Adaptation Prediction 프로젝트를 하며 수행한 EDA나 모델링, 모듈 등을 담고 있는 프로젝트 입니다.

## Getting Started



### Prerequisites

아나콘다 3.6 버전을 기준으로 작성되었으며, requirements.txt를 이용하여 다음과 같은 가상환경을 만들어 줍니다.

```
conda create --name <envname> --file requirements.txt
```

### How to Use pet_analyzer

#### 1. PetDataLoader
Kaggle 커널에서 데이터를 불러오는 클래스

* get_all_data(add_score=False, labels=False)

```
import pet_analyzer

pdl = PetDataLoader()
data = pdl.get_all_data() # type -> pandas.DataFrame
```
add_score : Magnitude, Score를 곱해 만든 새로운 파생변수인 SentimentScore를 생성한다.
labels : Breed, State, Color 라벨네임을 컬럼에 추가한다.

* preprocess_data(dataframe)
```
import pet_analyzer

pdl = PetDataLoader()
Not Ready
```

#### 2. AdoptionPredictor

여러 알고리즘을 이용하여 머신러닝 모델을 만드는 클래스.

* random_forest()

```
import pet_analyzer

ap = AdoptionPredictor()
# Not Ready
``` 
