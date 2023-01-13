import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam


def build_model(num_classes):
    # construct CNN structure
    model = keras.Sequential()
    # 1st convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3), padding='valid'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model


def train_test(trainx, trainy, testx, testy, val_x, val_y):
    model = build_model(5)
    model.summary()

    train_x, train_y, test_x, test_y = trainx, trainy, testx, testy
    train_x = train_x / 255
    train_y = train_y
    test_x = test_x / 255
    test_y = test_y
    print(train_y.shape)

    # Build CNN model with a learning rate of 1e-4
    learning_rate = 1e-4
    sgd_optimizer = Adam(learning_rate=learning_rate, decay=learning_rate / 20)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd_optimizer,
                  metrics=['accuracy'])

    # Train the CNN model with a batch_size of 100 and 50 cycles
    batch_size = 100
    epochs = 50
    model.fit(train_x, train_y,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(test_x, test_y))

    # Evaluate the accuracy and loss of the model
    loss, acc = model.evaluate(val_x, val_y)
    print(acc)
    print(loss)
