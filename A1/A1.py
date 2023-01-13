import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt


def allocate_weights_and_biases():
    # define number of hidden layers ..
    n_hidden_1 = 2048  # 1st layer number of neurons
    n_hidden_2 = 2048  # 2nd layer number of neurons
    n_hidden_3 = 2048
    n_hidden_4 = 2048

    # inputs placeholders
    X = tf.placeholder("float", [None, 68, 2])
    Y = tf.placeholder("float", [None, 2])  # 2 output classes

    # flatten image features into one vector (i.e. reshape image feature matrix into a vector)
    images_flat = tf.keras.layers.Flatten()(X)

    # weights and biases are initialized from a normal distribution with a specified standard devation stddev
    stddev = 0.01

    # define placeholders for weights and biases in the graph
    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([68 * 2, n_hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'hidden_layer3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=stddev)),
        'hidden_layer4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, 2], stddev=stddev))
    }

    biases = {
        'bias_layer1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([n_hidden_2], stddev=stddev)),
        'bias_layer3': tf.Variable(tf.random_normal([n_hidden_3], stddev=stddev)),
        'bias_layer4': tf.Variable(tf.random_normal([n_hidden_4], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
    }

    return weights, biases, X, Y, images_flat


def multilayer_perceptron():
    weights, biases, X, Y, images_flat = allocate_weights_and_biases()

    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(images_flat, weights['hidden_layer1']), biases['bias_layer1'])
    layer_1 = tf.sigmoid(layer_1)

    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['bias_layer2'])
    layer_2 = tf.sigmoid(layer_2)

    # Output fully connected layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer, X, Y


def A1_train_test(trainx, trainy, testx, testy, val_x, val_y):
    # learning parameters
    training_epochs = 1600
    learning_rate = 9e-6

    # display training accuracy every ..
    display_accuracy_step = 10

    training_images, training_labels, test_images, test_labels = trainx, trainy, testx, testy

    logits, X, Y = multilayer_perceptron()

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # define training graph operation
    train_op = optimizer.minimize(loss_op)

    # graph operation to initialize all variables
    init_op = tf.global_variables_initializer()
    epoch_list = np.zeros(training_epochs)
    cost_list = np.zeros(training_epochs)
    acc_list = np.zeros(training_epochs)
    inte = 0
    i = 0
    # run graph weights/biases initialization op
    with tf.Session() as sess:
        sess.run(init_op)
        # begin training loop ..
        for epoch in range(training_epochs):
            # complete code below
            # run optimization operation (backprop) and cost operation (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: training_images,
                                                               Y: training_labels})

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))
            epoch_list[inte] = int(epoch)
            cost_list[inte] = float(cost)
            inte += 1

            if epoch % display_accuracy_step == 0:
                pred = tf.nn.softmax(logits)  # Apply softmax to logits
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

                # calculate training accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Accuracy: {:.3f}".format(accuracy.eval({X: training_images, Y: training_labels})))
                acc_list[i] = accuracy.eval({X: training_images, Y: training_labels})
                i = i + 1

        print("Optimization Finished!")

        # -- Define and run test operation -- #

        # apply softmax to output logits
        pred = tf.nn.softmax(logits)

        #  derive inffered calasses as the class with the top value in the output density function
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Test Accuracy:", accuracy.eval({X: test_images, Y: test_labels}))
        plt.plot(epoch_list, cost_list)
        # plt.plot(epoch_list, acc_list)
        plt.legend(['loss', 'acc'])
        plt.grid()
        plt.show()
