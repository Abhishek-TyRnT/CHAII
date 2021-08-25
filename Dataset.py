import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import skipgrams


def preprocess_cntxt_to_sentences(file1, sentences_df, danda, destination_file_name):
    df_train = pd.read_csv(file1)
    string = 'abcdefghijklmnopqrstuvwxyz0123456789{}[]""!?():;,#*@-\/+=$%^&_<>'
    j = len(sentences_df)
    for (context,lang) in df_train[["context","language"]].values:
        context = context.lower()
        split_char = danda if lang == "hindi" else '.'
        context = context.replace('\n',split_char)
        for i in string:
            context = context.replace(i,"")
        for sentence in context.split(split_char):
            if sentence == " " or sentence == "":
                continue
            sentences_df.loc[j] = sentence

            j += 1

    sentences_df.to_csv(destination_file_name)


def get_char_dev(text):
    x, y = text[:4], text[4:]
    path = "data/Devanagri script .csv"
    devanagri = pd.read_csv(path)
    return devanagri.loc[devanagri['Unnamed: 0'] == x, y].values[0]


def generate_cooc_matrix(text, tokenizer, window_size, n_vocab, use_weighting=True):
    sequences = tokenizer.texts_to_sequences(text)
    n_vocab = len(tokenizer.word_index)
    cooc_mat = tf.zeros((n_vocab, n_vocab), dtype=np.float32)
    for sequence in sequences:
        for i, wi in zip(np.arange(window_size, len(sequence) - window_size), sequence[window_size:-window_size]):
            context_window = sequence[i - window_size: i + window_size + 1]
            distances = np.abs(np.arange(-window_size, window_size + 1))
            distances[window_size] = 1.0
            nom = np.ones(shape=(window_size * 2 + 1,), dtype=np.float32)
            nom[window_size] = 0.0

            if use_weighting:
                cooc_mat[wi, context_window] += nom / distances  # Update element
            else:
                cooc_mat[wi, context_window] += nom

    return cooc_mat

def generate_cooc_matrix_csv(file_path, window_size, n_vocab, dest_path,use_weighting=True):
    text = pd.read_csv(file_path)
    text = text['sentence'].values
    tokenizer = get_tokenizer(file_path, n_vocab)
    n_vocab = len(tokenizer.word_index.keys())
    cooc_mat = generate_cooc_matrix(text, tokenizer, window_size, n_vocab, use_weighting)
    cooc_mat = pd.DataFrame(cooc_mat)
    cooc_mat.to_csv(dest_path)


def get_tokenizer(file_path, v_size, unknown = 'UNK'):
    sentences_df = pd.read_csv(file_path)
    text = sentences_df['sentence']
    tokenizer = Tokenizer(num_words=v_size, oov_token=unknown)
    tokenizer.fit_on_texts(text.values)
    return tokenizer



def data_generator(path, path2):
    if isinstance(path, bytes):
        path = path.decode('utf-8')

    if isinstance(path2, bytes):
        path2 = path2.decode('utf-8')
    text = pd.read_csv(path)['sentence']

    v_size = 5000
    tokenizer = Tokenizer(num_words=v_size, oov_token='UNK')
    tokenizer.fit_on_texts(text.values)

    cooc_mat = pd.read_csv(path2).values
    for sentence in text:
        sequence = tokenizer.texts_to_sequences([sentence])[0]

        w_pairs, _ = skipgrams(sequence, vocabulary_size=v_size, negative_samples=0.0, shuffle=True)

        for sg_in, sg_out in w_pairs:
            X_ij = cooc_mat[sg_in, sg_out] + 1

            yield sg_in, sg_out, X_ij

file1 = 'data/train.csv'
danda = get_char_dev('u0964')
sentences = pd.read_csv('data/sentences.csv')


dest_path = 'data/sentences.csv'
tokenizer = get_tokenizer('data/sentences.csv', 200000)
generate_cooc_matrix_csv(dest_path, 4, 200000, 'data/cooc_mat.csv')


