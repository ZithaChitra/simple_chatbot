from util2 import loadPrepareData, trimRareWords, tagger
import pathlib
import os

corpus_name = "cornell movie-dialogs corpus"
corpus = pathlib.Path.cwd() / "Data" / corpus_name
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
MIN_COUNT = 3

voc, pairs = loadPrepareData(corpus, corpus_name, datafile)
keep_pairs, voc = trimRareWords(voc, pairs, MIN_COUNT)
# eos and sos tags
for pair in keep_pairs:
	pair[1] = tagger(pair[1])
	# pair = padding(pair, 10)
	
	
# Print some pairs to validate
print("\npairs:")
for pair in keep_pairs[:10]:
	print(pair)
