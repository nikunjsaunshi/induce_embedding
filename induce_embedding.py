import numpy as np


def split(text):
  '''Splits a text string into tokens by splitting on spaces
  Args:
    text: string
  Returns:
    list of strings
  '''
  return text.lower().split()


def load_vectors(vector_file):
  '''Loads word embeddings from a file
  Args:
    vector_file: string
  Returns:
    dict with key = word and value =  numpy array
  '''
  word_vector = {}
  with open(vector_file, 'r') as f:
    for line in f:
      index = line.index(' ')
      word = str(line[:index])
      vector = np.fromstring(line[index+1:], dtype=np.float32, sep=' ')
      word_vector[word] = vector
  return word_vector


def induce_embedding(target_word, context, word_vector, induction_matrix, window_size=5, tokenize_fn=split):
  '''Induces an embedding for a target word given a context of words. Computes
  context embedding as average of embeddings for words in the context in a
  window around the target word and multiplies this by the induction matrix.
  Args:
    target_word: string for a single word
    context: string for a sequence of words
    word_vector: dict with key = word and value = numpy array of shape (d,)
    induction_matrix: numpy array of shape (d,d)
    window_size: int
    tokenize_fn: function which splits a string to tokens
  Returns:
    numpy array of shape (d,)
  '''
  assert(len(word_vector))
  zero = np.zeros(list(word_vector.values())[0].shape[0])
  tokenized_context = tokenize_fn(context)
  if window_size is None:
    context_embedding = np.mean([word_vector.get(w, zero) for w in tokenized_context if w != target_word], 0)
  else:
    context_embedding = zero
    for i, word in enumerate(tokenized_context):
      if word == target_word:
        context_embedding += np.mean([word_vector.get(w, zero)
          for w in tokenized_context[max(0,i-window_size):i]+tokenized_context[i+1:i+window_size+1]], 0)
    context_embedding /= tokenized_context.count(target_word)
  return context_embedding.dot(induction_matrix)
  

def main():
  vector_file = './data/wikipedia_glove_300.txt'
  induction_file = './data/wikipedia_glove_300_induction.npy'

  word_vector = load_vectors(vector_file)
  induction_matrix = np.load(induction_file)
  vector = induce_embedding('freedom', 'Freedom of speech is amazing', word_vector, induction_matrix)


if __name__ == '__main__': main()
