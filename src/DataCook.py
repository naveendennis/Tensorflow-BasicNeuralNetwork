import numpy as np
import pandas as pd
import nltk
from tensorflow.contrib import learn
from sklearn.preprocessing import OneHotEncoder
import os.path


def get_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('src',
                                                  os.path.join(
                                                      os.path.dirname(os.path.realpath(__name__)),
                                                      'src',
                                                      'Utils.py'))
    src_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(src_module)
    return src_module

class DataCook:
    def __init__(self, file_name, features, labels, delimiter='\t'):
        self.features = features
        self.labels = labels
        self.headers = self.features+self.labels
        self.dataframe = self.__get_dataframe__(file_name, delimiter=delimiter)
        self.vocab_processor_cache = []
        self.input_vector, self.vocab_map = self.__get_selected_list__(selection_list=features,
                                                                       is_label=False)
        self.label_vector = self.__get_selected_list__(selection_list=labels)

    def get_input_vector(self):
        return self.input_vector

    def get_label_vector(self):
        return self.label_vector

    def __get_dataframe__(self, file_name, delimiter):
        return pd.read_csv(file_name, delimiter=delimiter, names=self.headers)

    def __get_selected_list__(self, selection_list, is_label=True):
        _dim = len(self.dataframe)
        result_vector = np.empty((_dim, 0))
        vocab_map = {}
        if len(selection_list) == 0:
            result_vector = self.dataframe
        else:
            for _each_key in selection_list:

                if len(self.dataframe[_each_key].unique()) > 0 and \
                                type(self.dataframe[_each_key].unique()[0]) is str:
                    # String values are processed depending on whether they are labels or not
                    if is_label:
                        labels = list(self.dataframe[_each_key].unique())
                        _inv_map = {_: index for index, _ in enumerate(labels)}
                        _enc = OneHotEncoder()
                        self.dataframe['encoded_label_' + _each_key] = self.dataframe[_each_key].apply(
                            lambda row: _inv_map[row]
                        )
                        _enc.fit(np.array([
                            [each] for each in self.dataframe['encoded_label_' + _each_key].as_matrix()]))
                        self.dataframe['encoded_label_' + _each_key] = self.dataframe['encoded_label_' + _each_key].\
                            apply(lambda row: _enc.transform(np.array([[row]])).data)
                        result_vector = np.concatenate(
                            (result_vector, self.dataframe['encoded_label_' + _each_key].as_matrix().reshape(_dim, 1)),
                            axis=1)

                    else:
                        self.dataframe['tokenized_sents'] = self.dataframe[_each_key].apply(
                            lambda row: nltk.word_tokenize(row))
                        self.dataframe['sents_length'] = self.dataframe['tokenized_sents'].apply(lambda row: len(row))
                        _max_document_length = self.dataframe['sents_length'].max() \
                            if self.dataframe['sents_length'].max() < 64 else 64
                        _vocab_processor = learn.preprocessing.VocabularyProcessor(_max_document_length)
                        _str_feature = np.array(
                            list(_vocab_processor.fit_transform(self.dataframe[_each_key].tolist())))
                        self.vocab_processor_cache.append(_vocab_processor)
                        result_vector = np.concatenate((result_vector, _str_feature), axis=1)
                else:
                    # Non string features directly added to the feature matrix
                    _otr_feature = np.array(self.dataframe[_each_key].tolist()).reshape(_dim, 1)
                    result_vector = np.concatenate((result_vector, _otr_feature), axis=1)

        return result_vector, vocab_map


if __name__ == '__main__':
    properties = get_module().Utils().initialize()
    header_names = properties["header_names"]
    obj = DataCook(
        file_name=os.path.join(
            os.path.dirname(os.path.realpath(__name__)),
            properties["data_location"],
            properties["filename"]),
        features=header_names,
        labels=header_names[2:],
        delimiter='\t'
    )
    X, y = (obj.get_input_vector(), obj.get_label_vector())
    print(X, y)
