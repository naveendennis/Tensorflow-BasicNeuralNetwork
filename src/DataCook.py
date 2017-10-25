import numpy as np
import pandas as pd
import nltk
from tensorflow.contrib import learn
from sklearn.preprocessing import OneHotEncoder

class DataCook:
    def __init__(self, file_name, features, labels, delimiter='\t'):
        self.features = features
        self.labels = labels
        self.headers = self.features+self.labels
        self.dataframe = self.get_dataframe(file_name, delimiter=delimiter)
        self.input_vector = self.get_selected_list(selection_list=features, isLabel=False)
        self.label_vector = self.get_selected_list(selection_list=labels)

    def get_dataframe(self, file_name, delimiter):
        return pd.read_csv(file_name, delimiter=delimiter, names=self.headers)

    def get_selected_list(self, selection_list, isLabel=True):
        dim = len(self.dataframe)
        X = np.empty((dim, 0))
        if len(selection_list) == 0:
            X = self.dataframe
        else:
            for each_key in selection_list:

                if len(self.dataframe[each_key].unique()) > 0 and \
                                type(self.dataframe[each_key].unique()[0]) is str:
                    # String values are processed depending on whether they are labels or not
                    if isLabel:
                        labels = list(self.dataframe[each_key].unique())
                        inv_map = {_: index for index, _ in enumerate(labels)}
                        vocab = {v: k for k, v in inv_map.items()}
                        enc = OneHotEncoder()
                        self.dataframe['encoded_label_' + each_key] = self.dataframe[each_key].apply(
                            lambda row: inv_map[row]
                        )
                        enc.fit(np.array([[each] for each in self.dataframe['encoded_label_' + each_key].as_matrix()]))
                        self.dataframe['encoded_label_' + each_key] = self.dataframe['encoded_label_' + each_key].apply(
                            lambda row: enc.transform(np.array([[row]])).data
                        )
                        X = np.concatenate((X,
                                            self.dataframe['encoded_label_' + each_key].as_matrix().reshape(dim, 1)
                                            ),
                                           axis=1)

                    else:
                        self.dataframe['tokenized_sents'] = self.dataframe[each_key].apply(lambda row: nltk.word_tokenize(row))
                        self.dataframe['sents_length'] = self.dataframe['tokenized_sents'].apply(lambda row: len(row))
                        max_document_length = self.dataframe['sents_length'].max()
                        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
                        str_feature = np.array(list(vocab_processor.fit_transform(self.dataframe[each_key].tolist())))
                        X = np.concatenate((X, str_feature), axis=1)
                else:
                    # Non string features directly added to the feature matrix
                    otr_feature = np.array(self.dataframe[each_key].tolist()).reshape(dim, 1)
                    X = np.concatenate((X, otr_feature), axis=1)

        return X


if __name__ == '__main__':
    header_names = ['ID', 'CONTENT', 'EMOTICON', 'CONFIDENCE']
    obj = DataCook(
        file_name='../data/test.txt',
                     features=header_names,
                     labels=header_names[2:],
                     delimiter='\t'
    )
    X, y = (obj.input_vector , obj.label_vector)
    print(X,y)
