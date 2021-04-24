from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import numpy as np
import unicodedata
import pathlib
import os
import re



corpus_name = "cornell movie-dialogs corpus"
corpus = pathlib.Path.cwd() / "Data" / corpus_name
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# Default word tokens
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token


class Voc:
	def __init__(self, name):
		self.name = name
		self.trimmed = False
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3 # PAD, SOS, EOS


	def addSentence(self, sentence):
		for word in sentence.split(" "):
			self.addWord(word)


	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

	
	# Remove words below a certain count threshold
	def trim(self, min_count):
		if self.trimmed:
			return
		self.trimmed = True
		
		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)


		print("keep words {}/{} = {:.4f}".format(len(keep_words),
				 len(self.word2count), (len(keep_words) / len(self.word2count))))
		

		# reinitialize dictionaries
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3 # Count default tokens

		for word in keep_words:
			self.addWord(word)



			
		
# ---------------- some data preprocessing -----------------

MAX_LENGTH = 10 # Max sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim and remove non-letter charectors
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	s = re.sub(r"\s+", r" ", s).strip()
	return s


# Read query/response pairs and return a Voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs



MIN_COUNT = 3    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs, voc


# Trim voc and pairs
# pairs = trimRareWords(voc, pairs, MIN_COUNT)



# ----------------- other_prerocessing ----------------
# from blog post: 
# https://towardsdatascience.com/how-to-implement-seq2seq-lstm-model-in-keras-shortcutnlp-6f355f3e5639


# put sos and eos tags for decoder input
def tagger(decoder_input_sentence):
  sos = "SOS "
  eos = " EOS"
  final_target = sos + decoder_input_sentence + eos 
  return final_target


def padding(sent_pair, MAX_LEN):
  
  sent_pair[0] = pad_sequences(sent_pair[0], maxlen=MAX_LEN, padding='post', truncating='post', value="PAD")
  sent_pair[1] = pad_sequences(sent_pair[1], maxlen=MAX_LEN, padding='post', truncating='post', value="PAD")
  
  return sent_pair



# We use pretrained word2vec model from glove
# Call Glove file from XX
# Create Embedding Matrix from our Vocabulary
# Create Embedding Layer

# GLOVE_DIR = path for glove.6B.100d.txt
def glove_100d_dictionary(glove_dir):
	embeddings_index = {}
	f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
	for line in f:
		values = line.split(" ")
		word = values[0]
		coeffs = np.asarray(values[1:], dtype="float32")
		embeddings_index[word] = coeffs
	f.close()
	return embeddings_index


# Create embeddings matrix from our vocabulary
# embeddings matrix: 100d
def embeddings_matrix_creator(
	word_index: dict,
	embeddings_index: dict,
	embedding_dim: int=100
):
	embeddings_matrix = np.zeros(len(word_index)+1, embedding_dim)
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will all be zero
			embeddings_matrix[i] = embedding_vector
	return embeddings_matrix


# Create embedding layer
def embedding_layer(
	num_words: int, embedding_dim: int, max_len: int, 
	embedding_matrix: dict
):
	embedding_layer = Embedding(
						input_dim=num_words,
						output_dim=embedding_dim,
						input_length=max_len,
						weights=embedding_matrix)
	return embedding_layer




if __name__ == "__main__":
	save_dir = os.path.join(corpus, "preprocessed_step_1.txt")
	voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
	keep_pairs, voc = trimRareWords(voc, pairs, MIN_COUNT)
	# eos and sos tags
	for pair in keep_pairs:
		pair[1] = tagger(pair[1])
		# pair = padding(pair, 10)
	
	
	# Print some pairs to validate
	print("\npairs:")
	for pair in keep_pairs[:10]:
		print(pair)

















		





			



