from __future__ import print_function


import keras
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


path = get_file(
    'tolkien-full.txt',
    origin='https://s3.ap-southeast-2.amazonaws.com/dataset-tolkien/tolkien-full.txt?response-content-disposition=inline&X-Amz-Security-Token=AgoGb3JpZ2luEPT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLXNvdXRoZWFzdC0yIoACFi40I%2FyFfXHQvK4Qt7kwZrLIMNfNnsqyoS1cyA%2FxUmMRg4%2FLpN6UFzy1rdzlaqCAOx0628zpGqZDjytxMcmPT0cQuwCkseQE9wXUZe5Yh7Fk8Zcg5s2zsMKO80P405RTGuuzBcWPVrjOQRec65GTqKSKWCDe0xiJ9NksKbp2h2%2F9yUmYsDuNZpz6bMKKjuCnmOnru3DXA76XH0WkmPzZP0Dh2iDYuAtzgmZOm8pNp9jgnVlSiPQzTCKHWkCVKCJFqU9nKjjEn%2Bqyy1tKTmvpaeR7uDtU8oRYh26tt7zeBMVzze6WN8aQfD9WVqjUG0qTRNuT%2Fgd3yzdyUBTJ%2F3DCyyrbAwgqEAAaDDk5MzkzOTAwODE1NSIMDM5hY2ka%2BytLKM1TKrgD07UMlGNaudNwJbKrRN8smw3vD8jESu1jIaHg6IMYMBMRbFyy5WegTYuXMJKeU2sl8z1XpXdrV0HFmo2Hzu5KoDVe6dN%2BzkCyYRYW1Nca4tf1YzG9l5oSWGpTJjfhvaOXAdEMhPgUnZurLTGQXlEOqplifto07kSuwqE4FNpiEfzZExAnBTleX8bMhFNUHaYbcwuMl8SAP0hzsSIGpIs868OQy2jVXE5SWWZO4SodEUklalHhZwhArrp2EN07Q0ZypX4E76bjGUVsNjf6XCKMDdsPad%2FxUNztaVsNeV60m5Jsc1cVLGV12G7VNwbeU95AqIY5OjTA9bJt633ugufXadeKGq46g4v3wMXR5HhIvzMBpXFyQeKkgb0gH8936%2BfR3o2sA6%2BEyUBj80dVpIOxOK8AJL9S0cX%2BLMWFoawKSV1GiJ6IyFhTnfL1goC3guhX0Myu9EjR%2BZhsBjxTTMwRj%2FHkOq%2BVbB0LBLZglxDAsOcwwsovptmBCK1Z0abbHnUbCH9vbeNnm1E0tG%2FbkI5GQO5n5qaYn%2Fqv22uS8PLF7QOdNEYhvyTT9CSYCQ3VWEDVd98oh9T33hQwzu%2FT4wU%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190226T083414Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIA6O23BVKNQ45IRGUQ%2F20190226%2Fap-southeast-2%2Fs3%2Faws4_request&X-Amz-Signature=a3b4386b17acb848b7f600ba1dad8e7d64c52e9f192f848721ab8e970bf11e63')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            console.log('hello')
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])