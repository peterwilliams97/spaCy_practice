"""
    # Install Ubuntu 16.0.4

    # Preliminaries
    sudo apt-get update
    sudo apt-get install zip
    sudo apt-get install git

    # Create our  directory structure
    cd ~
    mkdir downloads
    mkdir code
    mkdir data

    # Get ready to install stuff
    cd ~/downloads

    # GPUs
    # Ubuntu 16.0.4
    # Install Cuda drivers https://developer.nvidia.com/cuda-downloads
    # http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda

    # `cuda` is a symlink to the cuda directory which is versioned
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    cuda-install-samples-8.0.sh

    # Check which GPUs are on this machine
    cat /proc/driver/nvidia/version
    >NVRM version: NVIDIA UNIX x86_64 Kernel Module  375.66  Mon May  1 15:29:16 PDT 2017
    >GCC version:  gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4)

    # Check which are on this machine
    nvcc -V
    >nvcc: NVIDIA (R) Cuda compiler driver
    >Copyright (c) 2005-2016 NVIDIA Corporation
    >Built on Tue_Jan_10_13:22:03_CST_2017
    >Cuda compilation tools, release 8.0, V8.0.61

    # Install anaconda 3. https://www.continuum.io/downloads
    wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
    bash Anaconda3-4.4.0-Linux-x86_64.sh
    source ~/.bashrc  # Or open a new shell

    # Install tensorflow (gpu version)
    # http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html
    conda install tensorflow-gpu

    # Install keras
    conda install keras

    # Install spaCy on anaconda
    conda config --add channels conda-forge
    conda install spacy
    python -m spacy download en

    # Download data
    # Data is from http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip (
    #            in https://nlp.stanford.edu/sentiment/index.html)
    cd ~/data
    wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    unzip stanfordSentimentTreebank.zip


    # Install our test code
    cd ~/code

    python deep_learning_keras_peter.py -L 47 -i 200  -H 256  -n 40000 model.40000.256
    python deep_learning_keras_peter.py -L 47 -H 512 -i 100  model.40000.512.100 &

Done
    peter_williams@class-ubu-gpu:~/downloads$ ls -l /usr/local/
    total 36
    drwxr-xr-x  2 root root 4096 May 16 14:16 bin
    lrwxrwxrwx  1 root root    8 Jun  3 00:04 cuda -> cuda-8.0
    drwxr-xr-x 14 root root 4096 Jun  3 00:01 cuda-8.0

    peter_williams@class-ubu-gpu:~/downloads$ ls -l /usr/local/cuda/bin/
    total 55792
    -rwxr-xr-x 1 root root   137688 Jan 26 23:48 bin2c
    lrwxrwxrwx 1 root root        4 Jan 26 23:51 computeprof -> nvvp
    drwxr-xr-x 2 root root     4096 Jun  3 00:01 crt
    -rwxr-xr-x 1 root root  3951104 Jan 26 23:48 cudafe


   ???  https://unix.stackexchange.com/questions/218163/how-to-install-cuda-toolkit-7-x-or-8-on-debian-8-jessie-or-9-stretch
  ???  sudo apt-get install gcc g++ gcc-4.9 g++-4.9 libxi libxi6 libxi-dev libglu1-mesa libglu1-mesa-dev libxmu6 libxmu6-dev linux-headers-amd64 linux-source
"""

import plac
import os
import random

import pathlib
import cytoolz
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import CSVLogger

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import pickle

import spacy
import tensorflow as tf


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def np_show(name, o):
    try:
        print('%s: %s.%s' % (name, list(o.shape), o.dtype))
    except:
        try:
            print('%s: %s %d:%s' % (name, type(o), len(o), type(o[0])))
        except:
            print('%s: %s ***' % (name, type(o)))


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape,
                                     arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, np.asarray(labels, dtype='bool')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype='int32')
    max_feature_idx = -1
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space:
                Xs[i, j] = token.rank + 1
                j += 1
                max_feature_idx = max(j, max_feature_idx)
                if j >= max_length:
                    break
    print('******get_features: max_feature_idx=%d' % max_feature_idx)
    return Xs


def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100, epochs=5,
          by_sentence=True, callbacks=None):
    print("Loading spaCy")
    np_show("train_texts", train_texts)
    np_show("train_labels", train_labels)
    np_show("dev_texts", dev_texts)
    np_show("dev_labels", dev_labels)
    nlp = spacy.load('en', entity=False)
    np_show("nlp.vocab", nlp.vocab)
    embeddings = get_embeddings(nlp.vocab)
    np_show("embeddings", embeddings)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts, batch_size=5000, n_threads=3))
    dev_docs = list(nlp.pipe(dev_texts, batch_size=5000, n_threads=3))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)
    np_show("train_docs", train_docs)
    np_show("train_labels", train_labels)
    np_show("dev_docs", dev_docs)
    np_show("dev_labels", dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    np_show('train_X', train_X)
    np_show('dev_X', dev_X)
    print('train_X:', train_X[0, :])
    print('dev_X  :', dev_X[0, :])
    print('fit <<<<<')
    model.fit(train_X, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(dev_X, dev_labels),
              callbacks=callbacks
              # validation_split=0.5
              )
    print('fit >>>>>')
    return model


def compile_lstm(embeddings, shape, settings):

    model = Sequential()
    # with tf.device('/gpu:0'):
    model.add(Embedding(embeddings.shape[0],
                        embeddings.shape[1],
                        input_length=shape['max_length'],
                        trainable=False,
                        weights=[embeddings],
                        mask_zero=True))
    # with tf.device('/gpu:1'):
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    # with tf.device('/gpu:1'):
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                     dropout=settings['dropout'],
                                     recurrent_dropout=settings['dropout'])))
    # with tf.device('/gpu:3'):
    model.add(Dropout(0.1))  # !@#$ Get bool warning without this layer
    model.add(Dense(shape['nr_class'], activation='sigmoid'))

    # model = make_parallel(model, 4)

    # try using different optimizers and different optimizer configs
    model.compile(optimizer=Adam(lr=settings['lr']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )

    print("compile_lstm: shape=%s settings=%s" % (shape, settings))
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 dropout=settings['dropout'],
                                 recurrent_dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))

    model.compile(optimizer=Adam(lr=settings['lr']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_embeddings(vocab):
    max_rank = max(lex.rank + 1 for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    return vectors


def evaluate(model_dir, texts, labels, max_length=100):
    def create_pipeline(nlp):
        return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
                                                               max_length=max_length)]

    nlp = spacy.load('en')
    nlp.pipeline = create_pipeline(nlp)

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i


HOME = os.path.expanduser("~")
SENTIMENT_DIR = os.path.join(HOME, 'data/stanfordSentimentTreebank')


def get_data_dict(data_dir):
    id_phrase = {}
    path = os.path.join(data_dir, 'dictionary.txt')
    with open(path, 'rt') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            parts = line.split('|')
            assert len(parts) == 2, line
            id_ = int(parts[1])
            assert id_ not in id_phrase, (id_, line, id_phrase[id_])
            id_phrase[id_] = parts[0]
            # print('%3d: "%s"' % (i, line))
            # print('   ', len(parts), parts)
            # if i >= 6:
            #     break
    return id_phrase


def get_data_sentiment(data_dir):
    id_sentiment = {}
    path = os.path.join(data_dir, 'sentiment_labels.txt')
    with open(path, 'rt') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.rstrip('\n')
            parts = line.split('|')
            assert len(parts) == 2, line
            id_ = int(parts[0])
            sentiment = float(parts[1])
            assert id_ not in id_sentiment, (id_, line, id_sentiment[id_])
            id_sentiment[id_] = sentiment
            # print('%3d: "%s"' % (i, line))
            # print('   ', len(parts), parts)
            # if i >= 6:
            #     break
    return id_sentiment


DATA_TRAIN, DATA_DEV = 0, 1


has_read = False
id_text = None
id_sentiment = None
id_list = None


def read_data(data_set, limit=0):
    global has_read, id_text, id_sentiment, id_list
    print('@@@@@ read_data has_read=%s' % has_read)

    data_dir = SENTIMENT_DIR
    threshold = 0.5

    if not has_read:
        id_text = get_data_dict(data_dir)
        id_sentiment = get_data_sentiment(data_dir)
        id_list = list(id_text)
        # text_sentiment = {id_text[i]: id_sentiment[i] for i in id_text}
        random.seed(4)
        np_show("id_text", id_text)
        np_show("id_sentiment", id_sentiment)
        np_show("id_list", id_list)
        random.shuffle(id_list)

        has_read = True

    if limit >= 1:
        id_list = id_list[:limit]
        np_show("id_text", id_text)

    half = int(round(len(id_list) * 0.8))
    spans = {
        DATA_TRAIN: (0, half),
        DATA_DEV: (half, len(id_list)),
    }
    t0, t1 = spans[data_set]
    id_list = id_list[t0:t1]

    # pos = [(text, 1)for text in text_list if text_sentiment[text] >= threshold]
    # neg = [(text, 0) for text in text_list if text_sentiment[text] < threshold]
    texts = [id_text[i] for i in id_list]
    labels = [int(id_sentiment[i] > threshold) for i in id_list]
    np_show("texts", texts)
    np_show("labels", labels)
    n_pos = len([l for l in labels if l])
    n_neg = len([l for l in labels if not l])
    n_tot = n_pos + n_neg
    print('pos + neg: %d + %d = %d %.2f %.2f' % (n_pos, n_neg, n_tot, n_pos / n_tot, n_neg / n_tot))
    return texts, labels

    # examples = []
    # for subdir, label in (('pos', 1), ('neg', 0)):
    #     for filename in (data_dir / subdir).iterdir():
    #         with filename.open() as file_:
    #             text = file_.read()
    #         examples.append((text, label))
    # random.shuffle(examples)
    # if limit >= 1:
    #     examples = examples[:limit]
    # return zip(*examples)  # Unzips into two lists


@plac.annotations(
    model_dir=("Location of output model directory",),
    is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
    nr_hidden=("Number of hidden units", "option", "H", int),
    max_length=("Maximum sentence length", "option", "L", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    epochs=("Number of training epochs", "option", "i", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int)
)
def main(model_dir,
         is_runtime=False,
         nr_hidden=64, max_length=100,  # Shape
         dropout=0.5, learn_rate=0.001,  # General NN config
         epochs=5, batch_size=100, nr_examples=-1):  # Training params
    model_dir = pathlib.Path(model_dir)
    try:
        os.makedirs(str(model_dir))
    except FileExistsError:
        pass
    training_path = model_dir / 'training.log'
    with training_path.open('wt') as f:
        f.write('LOG')
    csv_logger = CSVLogger(str(training_path))
    callbacks = [csv_logger]

    # train_dir = pathlib.Path(train_dir)
    # dev_dir = pathlib.Path(dev_dir)
    if is_runtime:
        dev_texts, dev_labels = read_data(DATA_DEV)
        acc = evaluate(model_dir, dev_texts, dev_labels, max_length=max_length)
        print(acc)
    else:
        print("Read data")
        train_texts, train_labels = read_data(DATA_TRAIN, limit=nr_examples)
        dev_texts, dev_labels = read_data(DATA_DEV, limit=nr_examples)
        train_labels = np.asarray(train_labels, dtype=np.int64)
        dev_labels = np.asarray(dev_labels, dtype=np.int64)
        lstm = train(train_texts, train_labels, dev_texts, dev_labels,
                     {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
                     {'dropout': dropout, 'lr': learn_rate},
                     {},
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        weights = lstm.get_weights()

        with (model_dir / 'model').open('wb') as file_:
            pickle.dump(weights[1:], file_)
        with (model_dir / 'config.json').open('wt') as file_:
            file_.write(lstm.to_json())


if __name__ == '__main__':
    plac.call(main)
