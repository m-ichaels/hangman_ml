import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tqdm import tqdm

# sourcing the training word file and filling an empty dictionary with its contents
with open('words_250000_train.txt', 'r') as file:
    word_dict = [line.strip() for line in file]

# due to time/processing constraints a subset of the training words were used to train the model. Random sampling was used, hoping to gain a subset reasonably representative of the parent dataset
sample_size = 100

# a repeatable random selection is used for reproducability
np.random.seed(42)
# shuffling words to avoid bias via training order.
words = np.random.permutation(list(word_dict))[:sample_size]

# defines the parameters for the machine learning setup
max_length = max(len(word) for word in word_dict)
epoch_size = sample_size
num_epochs = 1
batch_size = int(np.array([len(i) for i in words[:sample_size]]).mean())

class HangmanPlayer:
    def __init__(self, word, model, lives=6):
        # initialising variables, sets, and lists for future use
        self.target = word
        self.target_num = [ord(i) - 97 for i in word]
        self.letters_guessed = set()
        self.letters_remaining = set(self.target_num)
        self.lives_remaining = lives
        self.states_history = []
        self.letters_previously_guessed = []
        self.guesses = []
        self.correct_guesses = []
        self.model = model

    def target_word_to_num(self):
        # turning the target word into a one-hot encoded representation of the word
        word = [i if i in self.letters_guessed else 26 for i in self.target_num]
        target_data = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            target_data[i, j] = 1

        # Padding the data sequence to max_length length with zeros
        if len(target_data) < max_length:
            padding = np.zeros((max_length - len(target_data), 27), dtype=np.float32)
            target_data = np.vstack([target_data, padding])
        # The following should not occur, just coded in the instance a word is used that is longer that the longest word in the training set (the word is truncated)
        elif len(target_data) > max_length:
            target_data = target_data[:max_length]
        return target_data

    def guess_to_num(self, guess):
        # encoding the guessed letter
        guess_num = np.zeros(26, dtype=np.float32)
        guess_num[guess] = 1
        return guess_num

    def previous_guesses_to_num(self):
        # encoding the previously guessed lettered
        guess = np.zeros(26, dtype=np.float32)
        for i in self.letters_guessed:
            guess[i] = 1
        return guess

    def correct_guesses_to_num(self):
        # encoding correct responses
        response = np.zeros(26, dtype=np.float32)
        for i in self.letters_remaining:
            response[i] = 1.0
        response /= response.sum()
        return response

    def store_info(self, guess):
        # manages the game info
        # stores the current state of the game
        self.states_history.append(self.target_word_to_num())
        self.letters_previously_guessed.append(self.previous_guesses_to_num())

        # updates list of guesses and guessed letters
        self.guesses.append(guess)
        self.letters_guessed.add(guess)

        # updates correct guesses
        correct_guesses = self.correct_guesses_to_num()
        self.correct_guesses.append(correct_guesses)

        # removes guessed letters from allowed guesses
        if guess in self.letters_remaining:
            self.letters_remaining.remove(guess)

        # deducts a life for an incorrect guess
        if self.correct_guesses[-1][guess] < 0.00001:
            self.lives_remaining -= 1

    def run(self):
        # executes the game
        # predicts a letter when the user has lives remaining and letters still to guess
        while self.lives_remaining > 0 and len(self.letters_remaining) > 0:
            inputs = [
                np.expand_dims(self.target_word_to_num(), axis=0),
                np.expand_dims(self.previous_guesses_to_num(), axis=0)
            ]
            prediction = self.model.predict(inputs)
            guess = np.argmax(prediction)
            self.store_info(guess)

        # adds new info to its required variables
        return (
            np.array(self.states_history),
            np.array(self.letters_previously_guessed),
            np.array(self.correct_guesses)
        )

# defining model inputs
input_hidden_word = Input(shape=(max_length, 27), name='input_hidden_word')
input_previous_guesses = Input(shape=(26,), name='input_previous_guesses')

# defining LSTM layer
lstm = LSTM(32, return_sequences=False)(input_hidden_word)
# concatenation of inputs
inputs_together = Concatenate()([lstm, input_previous_guesses])
# dense output layer
dense_out = Dense(26, activation='softmax')(inputs_together)

# model compilation
model = Model(inputs=[input_hidden_word, input_previous_guesses], outputs=dense_out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def pad_sequence(sequence, max_length):
    # pad sequences to same length (if not same length)
    padded = np.zeros((max_length, sequence.shape[1]), dtype=np.float32)
    padded[:min(len(sequence), max_length), :] = sequence[:max_length]
    return padded

trained_examples = 0
for epoch in range(num_epochs):
    # iterates through each epoch
    i = 0
    with tqdm(total=epoch_size, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        while trained_examples < (epoch + 1) * epoch_size:
            # runs through the words list
            word = words[i]
            i += 1

            # creates an instance of the HangmanPlayer class, and runs the game
            agent = HangmanPlayer(word, model)
            words_seen, previous_letters, correct_guesses = agent.run()

            # generates the following data into padded numpy arrays
            padded_nums_seen = np.array([pad_sequence(seq, max_length) for seq in words_seen])
            previous_letters = np.array(previous_letters)
            correct_guesses = np.array(correct_guesses)

            # trains the model based on the gamae and data iteration generated above
            model.train_on_batch([padded_nums_seen, previous_letters], correct_guesses)
            trained_examples += 1
            pbar.update(1)

# saves the model
model_filename = './hangman.keras' # file should either be .h5 or .keras, either is usable
model.save(model_filename)