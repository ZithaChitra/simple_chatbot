import tensorflow as tf
import pathlib
from zipfile import ZipFile
# import pandas as pd
import os



text_path = pathlib.Path.cwd() / "Data" 

def get_data(download_dir, file_name="movie_data"):
	"""
	Download Cornell Movie — Dialogs Corpus Dataset
	which contains over 220,579 conversational exchanges
	between 10,292 pairs of movie characters. And it 
	involves 9,035 characters from 617 movies.
	Parameters:
	----------
	download: Path
		Absolute path to download zip file to
	file_name: str
		Name of zipfile
	"""
	file_path = os.path.join(download_dir, file_name)
	_ = tf.keras.utils.get_file(
						file_path,
						"http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
						untar=True,
						archive_format="auto"
						)
	filename = f"{download_dir}/{file_name}.tar.gz"
	with ZipFile(filename, "r") as zip:
		zip.printdir()
		print("Extracting all files now..")
		zip.extractall(download_dir)
		print("Done!")



# get_data(text_path)
# movie_t = os.path.join(path, "cornell_movies", "movie_titles_metadata.txt")
movie_line_dir = os.path.join(text_path, "cornell movie-dialogs corpus", "movie_lines.txt")
movie_conv_dir = os.path.join(text_path, "cornell movie-dialogs corpus", "movie_conversations.txt")


		
