import sys
import regex as re
import pickle
import os
import pandas as pd
import numpy as np

word_index = {}

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def index_one_file(dir,file_name):
    text = open(dir+"/"+file_name).read()
    iter_in_text = re.finditer(r'[a-zA-Zäåö]+', text.lower())

    while 1:
        try:
            next_match = next(iter_in_text)
        except StopIteration:
            break
        else:
            word = next_match.group()
            position = next_match.start()
            if word in word_index:
                file_index = word_index[word]
                if file_name in file_index:
                    file_index[file_name].append(position)
                else:
                    file_index[file_name] = [position]
            else:
                word_index[word] = {}
                word_index[word][file_name] = [position]

def tf_idf_matrix():
    word_df = pd.DataFrame(word_index)
    word_non_na = word_df.replace(np.nan, "")
    len_word = word_non_na.applymap(len)
    word_count_book = len_word.apply(np.sum, axis=1)
    book_count_word = (len_word != 0).sum(0)
    book_number = len(word_count_book)
    word_tf_df = len_word.div(word_count_book, axis=0)
    word_idf_ser = np.log10(book_number / book_count_word)
    word_tf_idf = word_tf_df.mul(word_idf_ser, axis=1)
    return word_tf_idf

def cos_similarity(df):
    similarity = {}
    book_names = df.index
    book_number = len(book_names)
    for i in range(book_number):
        for j in range(i, book_number):
            name_a = book_names[i]
            name_b = book_names[j]
            book_a = df.loc[name_a, :]
            book_b = df.loc[name_b, :]
            ab_similarity = (book_a.dot(book_b)) / (np.sqrt(np.square(book_a).sum()) * np.sqrt(np.square(book_b).sum()))
            if name_a in similarity:
                similarity[name_a][name_b] = ab_similarity
            else:
                similarity[name_a] = {name_b: ab_similarity}
    return pd.DataFrame(similarity)

if __name__ == "__main__":
    # pickle.dump(index_one_file(sys.argv[1]), open(sys.argv[1]+".idx", "wb"))
    files = get_files("Selma","txt")
    for file in files:
        index_one_file("Selma",file)
    # pickle.dump(word_index, open("index.idx", "wb"))
    df = tf_idf_matrix()
    # print(df[["känna", "gås", "nils", "et"]])
    print(cos_similarity(df))


