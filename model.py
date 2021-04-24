from tensorflow.keras import Model
from tensorflow.keras import layers


def embed_layer():
	pass


def seq2seq_model_builder(
	num_words: int,
	word_dim: int=50,
	max_len: int=20,
	hidden_units: int=300,
)->Model:
	
	encoder_inputs = layers.Input(shape=(max_len,), dtype="int32")
	encoder_embedding = layers.Embedding(input_dim=num_words,
							 output_dim=word_dim)(encoder_inputs)
	encoder_lstm1 = layers.LSTM(hidden_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder_lstm1(encoder_embedding)


	decoder_inputs = layers.Input(shape=(max_len,), dtype="int32")
	decoder_embedding = layers.Embedding(input_dim=num_words,
					 		output_dim=word_dim)(decoder_inputs)
	decoder_lstm1 = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm1(decoder_embedding, initial_state=[state_h, state_c])

	
	# dense_layer = layers.Dense(num_words, activation="softmax")
	outputs = layers.TimeDistributed(layers.Dense(num_words,
						 activation="softmax"))(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], outputs)
	return model



if __name__ == "__main__":
	model = seq2seq_model_builder(400)
	model.summary()



