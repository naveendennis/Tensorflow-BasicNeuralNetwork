import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from tensorflow.contrib import learn
from sklearn.preprocessing import OneHotEncoder


def get_selected_list(dataframe, selection_list=[], isLabel=True):
    dim = len(dataframe)
    X = np.empty((dim, 0))
    if len(selection_list) == 0:
        X = dataframe
    else:
        for each_key in selection_list:

            if len(dataframe[each_key].unique()) > 0 and \
                            type(dataframe[each_key].unique()[0]) is str:
                # String values are processed depending on whether they are labels or not
                if isLabel:
                    labels = list(dataframe[each_key].unique())
                    inv_map = {_: index for index, _ in enumerate(labels)}
                    vocab = {v: k for k, v in inv_map.items()}
                    enc = OneHotEncoder()
                    dataframe['encoded_label_'+each_key] = dataframe.apply(
                        lambda row: inv_map[row[each_key]],
                        axis=1
                    )
                    enc.fit(np.array([[each] for each in dataframe['encoded_label_' + each_key].as_matrix()]))
                    dataframe['encoded_label_'+each_key] = dataframe.apply(
                        lambda row: enc.transform(row['encoded_label_'+each_key]),
                        axis=1
                    )
                    X = np.concatenate((X,
                                        dataframe['encoded_label_' + each_key].as_matrix().reshape(dim, 1)
                                        ),
                                       axis=1)

                else:
                    dataframe['tokenized_sents'] = dataframe.apply(lambda row:
                                                                   nltk.word_tokenize(row[each_key]),
                                                                   axis=1)
                    dataframe['sents_length'] = dataframe.apply(lambda row:
                                                                len(row['tokenized_sents']),
                                                                axis=1)
                    max_document_length = dataframe['sents_length'].max()
                    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
                    str_feature = np.array(list(vocab_processor.fit_transform(dataframe[each_key].tolist())))
                    X = np.concatenate((X, str_feature), axis=1)
            else:
                # Non string features directly added to the feature matrix
                otr_feature = np.array(dataframe[each_key].tolist()).reshape(dim, 1)
                X = np.concatenate((X, otr_feature), axis=1)

    return X


header_names = ['ID', 'CONTENT', 'EMOTICON', 'CONFIDENCE']


def read_data(file_name, headers=[], labels=[], features=[], delimiter=','):
    df = pd.read_csv(file_name, delimiter=delimiter, names=headers)
    X = get_selected_list(dataframe=df, selection_list=features, isLabel=False)
    y = get_selected_list(dataframe=df, selection_list=labels)
    return X, y


X, y = read_data('data/test.txt',
                 headers=header_names,
                 labels=header_names[2:],
                 features=[header_names[1]],
                 delimiter='\t')
