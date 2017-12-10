import autoEncoder
import sklearn.preprocessing as prep
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = autoEncoder.np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
ae = autoEncoder.AdditiveGaussianNoiseAutoenconder(n_input=784, n_hidden=200,
                                                   transfer_function=autoEncoder.tf.nn.softplus,
                                                   optimizer=autoEncoder.tf.train.AdamOptimizer(learning_rate=0.001),
                                                   scale=0.01)
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = ae.partical_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost:"+str(ae.cal_total_cost(X_test)))
