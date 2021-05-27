import tensorflow as tf
import matplotlib.pyplot as plt

imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

print(len(train_data[0]))
print(len(train_data[1]))

print(train_data.shape)



word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])



decode_review(train_data[0])


train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                         value=word_index['<PAD>'],
                                                         padding='post',
                                                         maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

print(len(train_data[0]))
print(len(train_data[1]))
print(train_data[0])



tf.random.set_seed(42)

vocab_size = 10000

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, 16),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
partial_y_train = train_labels[10000:]


results = model.evaluate(test_data, test_labels)
print(results)

def show_model_results(history):
  plt.plot(history.history['accuracy'], label='accuracy', c='bo')
  plt.plot(history.history['val_accuracy'], label='val_accuracy', c='b')
  plt.plot(history.history['loss'], label='loss', c='ro')
  plt.plot(history.history['val_loss'], label='val_loss', c='r')
  
 show_model_results(history=history)


random_review = np.random.randint(10000, size=256)   # Create a random review of size 256
biased_review = np.full(256, 530)   # Create a review of size 256 with only the word 'Brilliant'
test_data = np.append(test_data, [random_review], axis=0)   # Append the test dataset with the random review
test_data = np.append(test_data, [biased_review], axis=0)   # Append the test dataset with the biased review

print(model.predict(test_data))   # Predict on teh entire test dataset
print(model.predict(test_data[-2]))   # Print the predicted result of the random review
print(model.predict(test_data[-1])).  # Print the predicted result of the biased review

