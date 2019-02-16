# Kaggle Pet Adaptation Prediction

Pet Adaptation Prediction 프로젝트를 하며 수행한 EDA나 모델링, 모듈 등을 담고 있는 프로젝트 입니다.

## Getting Started



### Prerequisites

아나콘다 3.6 버전을 기준으로 작성되었으며, requirements.txt를 이용하여 다음과 같은 가상환경을 만들어 줍니다.

```
conda create --name <envname> --file requirements.txt
```

### How to Use pet_analyzer

#### 1. ExtractSentiment
train_sentiment, test_sentiment 디렉터리의 json 파일을 읽어들여 sentiment와 score를 추출하는 클래스

##### 1) get_sentiment(self, path)
sentiment 정보를 가져와서 데이터 프레임으로 만드는 함수

```
import pet_analyzer

train_sentiment_path = "../input/train_sentiment"
test_sentiment_path = "../input/test_sentiment"

es = ExtractSentiment()

train_sentiment = es.get_sentiment(train_sentiment_path) # type -> pandas.DataFrame
test_sentiment = es.get_sentiment(test_sentiment_path) # type -> pandas.DataFrame
```

#### 2. PetDataLoader
Kaggle 커널에서 데이터를 불러오는 클래스

##### 1) get_all_data(sentiment=True, add_score=False, labels=False, bin_fee=False)
train, test, sentiment 데이터를 로드한 후, 가공해서 데이터프레임으로 리턴하는 함수.
```
import pet_analyzer

pdl = PetDataLoader()
data = pdl.get_all_data() # type -> pandas.DataFrame
```
* sentiment : 감성 데이터를 merge하여 함께 불러온다.
* add_score : Magnitude, Score를 곱해 만든 새로운 파생변수인 SentimentScore를 생성한다.
* labels : Breed, State, Color 라벨네임을 컬럼에 추가한다.
* bin_fee : Fee 컬럼을 무료분양인 경우와 아닌경우 두 가지로 인코딩한다. (구현 예정)


#### 3. AdoptionPredictor

여러 알고리즘을 이용하여 머신러닝 모델을 만드는 클래스.

##### 1) boosting(use_xgb=True, submission=False, validation=True)
부스팅 알고리즘을 이용하여 모델을 만드는 함수.
```
import pet_analyzer

pdl = PetDataLoader()
data = pdl.get_all_data()

# 인스턴스 초기화 시, PetDataLoader를 이용해 가져온 데이터프레임을 파라미터로 넣어준다.
ap = AdoptionPredictor(data)

# boosting 모델을 불러올 때
model = ap.boosting(submission=False)

# submission하기 위해 test 데이터를 예측한 결과를 리턴
pred = ap.boosting(submission=True)
``` 

* use_xgb : True인 경우 xgboost 라이브러리를 사용하여 모델을 만든다.
* submission : True인 경우 test데이터를 예측한 결과를 리턴하고, False인 경우에는 만들어진 모델을 리턴한다.
* validation : True인 경우 validate_model() 함수를 이용하여 Cross Validation으로 모델의 정확도를 출력한다.

##### 2) validate_model(model, val_type="cv", X_test=None, y_test=None)
모델의 정확도를 확인하는 함수.

```
import pet_analyzer

ap = AdoptionPredictor()
# boosting 모델을 불러올 때
boosting_model = ap.boosting(submission=False)

# Cross Validation을 이용하여 모델의 정확도를 측정한다.
ap.validate_model(model=boosting_model, val_type="cv")

``` 

* model : 평가할 모델
* val_type : 모델의 Accuracy를 측정하는 방법을 결정한다.. "cv" -> cross_validation, "" -> 일반적인 Accuracy 측정
* X_test : 테스트 데이터로 사용할 집합.
* y_test : 테스트 데이터의 타겟 집합.