import collections
import math
import random
import zipfile
import numpy as np
import tensorflow as tf


def read_data():
    with zipfile.ZipFile('word2vecData/text8.zip') as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data()

vocabulary_size = 50000


def build_datasets(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_datasets(words)
del words
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 1 + 2 * skip_window
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 1

valid_size = 16
valid_window = 100
valid_example = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_datasets = tf.constant(valid_example, dtype=tf.int32)

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_bias = tf.zeros([vocabulary_size])

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_bias,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_datasets)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as sess:
    init.run()
    print('initialized')

    avg_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        avg_loss += loss_val
        if step % 2000 == 0:
            if step > 0:
                avg_loss /= 2000
            print('avg loss at step ', step, ' is ', avg_loss)
            avg_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_example[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                print(nearest)
                long_str = "Nearest to %s: " % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    long_str = "%s %s" % (long_str, close_word)
                print(long_str)

    final_embeddings = normalized_embeddings.eval()

import  matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs,labels,filename='wordVec.png'):
    assert low_dim_embs.shape[0]>=len(labels),'vec len err'
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
        plt.savefig(filename)

from sklearn.manifold import TSNE
tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot=100
low_dim_embs=tsne.fit_transform(final_embeddings[:plot,:])
label=[reverse_dictionary[i] for i in range(plot)]
plot_with_labels(low_dim_embs,label)
