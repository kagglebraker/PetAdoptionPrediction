import json
import os

import pandas as pd


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
        df.to_csv(path + "sentiment_test.csv", encoding='utf-8', index=False)

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

    def get_all_data(self, add_score=False, labels=False):
        """

        :param add_score: add Sentiment Score. default : False
        :return: all merged data (train+test+sentiment)
        """
        train = pd.read_csv(self.train_path)
        train = train.assign(is_train=True)

        test = pd.read_csv(self.test_path)
        test = test.assign(is_train=False)
        test = test.assign(AdoptionSpeed=9)

        sentiment = self.get_train_sentiment().append(self.get_test_sentiment())

        data = pd.concat([train, test], sort=True)

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


# class AdoptionPredictor(PetDataLoader):
#
#     def __init__(self):
#         # super().__init__()
#         pass
#
#     def one_hot_encoding(self):
#         pass
#
#     # pca, rf
#     def preprocess_pca(self):
#         pass
#
#     def random_forest(self):
#         pass
#
#     def decision_tree(self):
#         pass
#
#     def gradient_boosting(self, params):
#         pass


if __name__ == "__main__":
    pa = PetDataLoader()
    # d = pa.get_sentiment_data(merge_type="all")
    d = pa.get_all_data(add_score=False, labels=True)
    print(d.columns)
    print(d)
