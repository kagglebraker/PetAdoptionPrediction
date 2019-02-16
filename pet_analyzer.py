import json
import os
import xgboost as xgb
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
# import autosklearn.classification

class ParameterMismatchError(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class ExtractSentiment:
    def __init__(self):
        pass

    def get_sentiment_from_file(self, path, file_name):
        score = 99999
        magnitude = 99999
        with open(path + '/' + file_name, encoding='utf-8') as json_file:
            data = json.loads(json_file.read())
            pet_id = file_name.split(".")[0]
            magnitude = data["documentSentiment"]["magnitude"]
            score = data["documentSentiment"]["score"]

        return pet_id, magnitude, score

    def get_sentiment(self, path):
        file_list = os.listdir(path)
        rows_list = list()
        for file_name in file_list:
            pet_id, magnitude, score = self.get_sentiment_from_file(path, file_name)
            row = {"PetID": pet_id, "Score": score, "Magnitude": magnitude}
            rows_list.append(row)

        df = pd.DataFrame(rows_list)
        # df.to_csv(path + "sentiment_test.csv", encoding='utf-8', index=False)

        return df


class PetDataLoader(ExtractSentiment):

    def __init__(self):
        # super().__init__()
        self.base = "../input"
        self.train_path = self.base + "/train/train.csv"
        self.test_path = self.base + "/test/test.csv"
        self.train_sent_path = self.base + "/train_sentiment"
        self.test_sent_path = self.base + "/test_sentiment"

        train = pd.read_csv(self.train_path)
        self.target = train["AdoptionSpeed"]

        train = train.assign(is_train=True)
        # self.train = train.drop(labels=['AdoptionSpeed'], axis=1)
        self.train = train

        test = pd.read_csv(self.test_path)
        test = test.assign(is_train=False)
        test = test.assign(AdoptionSpeed=-1)
        self.test = test

    def get_train(self) -> pd.DataFrame:
        return self.train

    def get_test(self) -> pd.DataFrame:
        return self.test

    def get_train_sentiment(self) -> pd.DataFrame:
        train_sent = self.get_sentiment(self.train_sent_path)

        return train_sent

    def get_test_sentiment(self) -> pd.DataFrame:
        test_sent = self.get_sentiment(self.test_sent_path)
        return test_sent

    def get_all_data(self, sentiment=True, add_score=False, labels=False, bin_fee=False):
        """

        :param add_score: add Sentiment Score. default : False
        :return: all merged data (train+test+sentiment)
        """
        train = pd.read_csv(self.train_path)
        train = train.assign(is_train=True)

        test = pd.read_csv(self.test_path)
        test = test.assign(is_train=False)
        test = test.assign(AdoptionSpeed=9)

        data = pd.concat([train, test], sort=True)

        if sentiment:
            sentiment = self.get_train_sentiment().append(self.get_test_sentiment())
            data = pd.merge(data, sentiment, how="left", on="PetID")
            data = data.fillna(0)
            if add_score:
                data['SentimentScore'] = data.Magnitude * data.Score





        if labels:
            color_labels = pd.read_csv(self.base + "/color_labels.csv")
            breed_labels = pd.read_csv(self.base + "/breed_labels.csv")
            state_labels = pd.read_csv(self.base + "/state_labels.csv")

            data_color1 = data.merge(color_labels, how='left', left_on='Color1', right_on='ColorID', left_index=False,
                                     right_index=False).drop(columns='ColorID')
            data_color2 = data_color1.merge(color_labels, how='left', left_on='Color2', right_on='ColorID').drop(
                columns='ColorID')
            data_color3 = data_color2.merge(color_labels, how='left', left_on='Color3', right_on='ColorID').drop(
                columns='ColorID')

            data = data_color3

            data_breed1 = data.merge(breed_labels, how='left', left_on=['Breed1', 'Type'],
                                     right_on=['BreedID', 'Type']).drop(columns='BreedID')
            data_breed2 = data_breed1.merge(breed_labels, how='left', left_on=['Breed2', 'Type'],
                                            right_on=['BreedID', 'Type']).drop(columns='BreedID')
            data = data_breed2

            data_state = data.merge(state_labels, how='left', left_on='State', right_on='StateID').drop(
                columns='StateID')
            data = data_state

            mapping = {
                'ColorName_x': 'ColorName1',
                'ColorName_y': 'ColorName2',
                'ColorName': 'ColorName3',
                'BreedName_x': 'BreedName1',
                'BreedName_y': 'BreedName2'
            }
            data.columns = [mapping[colname] if colname in mapping.keys() else colname for colname in data.columns]

        return data


class AdoptionPredictor:

    def __init__(self, data):
        available_feature = ["AdoptionSpeed", "Age", "Breed1", "Breed2", "Color1", "Color2", "Color3", "Dewormed",
                             "Fee", "FurLength", "Gender", "Health", 'MaturitySize', "PhotoAmt", "Quantity", "State",
                             "Sterilized", "Type", "Vaccinated", "VideoAmt", "is_train", "Magnitude", "Score",
                             "SentimentScore"]

        # data = data.loc(data.columns.isin([available_feature]))

        data = data.drop(labels=["PetID", "RescuerID", "Description", "Name"], axis=1)

        train = data[data['is_train'] == True]
        self.target = train["AdoptionSpeed"]
        self.train = train.drop(labels=["AdoptionSpeed", "is_train"], axis=1)

        test = data[data['is_train'] == False]
        self.test = test.drop(labels=["AdoptionSpeed", "is_train"], axis=1)

    # pca, rf
    def preprocess_pca(self):
        pass

    def random_forest(self):
        # clf = RandomForestClassifier()
        pass

    def decision_tree(self):
        pass

    def boosting(self, use_xgb=True, submission=False, validation=True):
        test = self.test
        if submission:
            train = self.train
            target = self.target # train target
            X_test = None
            y_test = None
        else:
            train, X_test, target, y_test = train_test_split(self.train, self.target,
                                                                test_size=0.3,
                                                                random_state=42)

        if use_xgb:
            print("#" * 50)
            print("Gradient Boosting using xgboost")
            print("#" * 50)

            xgb_classifier = XGBClassifier()
            fitted_model = xgb_classifier.fit(train, target)
        else:
            model = GradientBoostingClassifier()
            fitted_model = model.fit(train, target)

        if validation:
            self.validate_model(fitted_model, val_type="cv", X_test=X_test, y_test=y_test)

        if submission:
            return fitted_model.predict(test)
        else:
            return fitted_model

    # model 넣고 train 정확도 출력
    def validate_model(self, model, val_type="cv", X_test=None, y_test=None):
        if not val_type in ["cv", ""]:
            raise ParameterMismatchError("val type is not exist")

        if X_test is None or y_test is None:
            train = self.train
            target = self.target
        else:
            train = X_test
            target = y_test

        print("#" * 50)
        print("Validation Start")
        print("#" * 50)

        if val_type == "cv":
            scores = cross_val_score(model, train, target, cv=5)
            print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
        else:
            predict = model.predict(train)
            score = accuracy_score(target, predict)

            print("Accuracy: %0.5f " % score)

    # def auto_sklearn(self):
    #     cls = autosklearn.classification.AutoSklearnClassifier()
    #     cls.fit(self.train, self.target)
    #     predictions = cls.predict(self.test)
    #
    #     return predictions


if __name__ == "__main__":
    pa = PetDataLoader()
    # Load Data
    d = pa.get_all_data(sentiment=False, add_score=False, labels=False)

    # Instance Init
    ap = AdoptionPredictor(d)
    print(ap.train.head())

    # get predicted class using boosting algorithm
    pred = ap.boosting(submission=True)
    # pred = ap.auto_sklearn()

    # make dataframe for submission
    submission = pd.DataFrame({'PetID': d[d["is_train"] == False].PetID, 'AdoptionSpeed': pred})
    submission.head()

    # submission
    submission.to_csv('submission.csv', index=False)
