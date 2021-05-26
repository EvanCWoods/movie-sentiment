import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import matplotlib.pyplot as plt 


def plot_graphs(history, metic):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabe(metric)
    plt.legend([metric, 'val_'+metric])



dataset, info = tfds.load('imdb_reviews', with_info=True,
                            as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset.element_spec

for example, label in train_dataset.take(1):
    print('text: ', example.numpy())
    print('label: ', label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for example, label in train_dataset.take(1):
    print('texts: ', example.numpy()[:3])
    print()
    print('labels: ', label.numpy()[:3])


VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

encoded_example = encoder(example)[:3].numpy()
encoded_example

for n in range(3):
    print('Original: ', example[n].numpy())
    print('Round-trip: ', ' '.join(vocab[encoded_example[n]]))
    print()

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=(encoder.get_vocabulary()),
        output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

print([layer.supports_masking for layer in model.layers])

sample_text = ['The movie was cool. The animation and the graphics '
                'were out of this world. I would recomend this movie.']
predictions = model.predict(np.array([sample_text]))
print(predictions[0])

padding = 'the ' * 2000
predictins = model.predict(np.array([sample_text, padding]))
print(predictions[0])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                metrics=['accuracy'])

history= model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

