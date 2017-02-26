#! /bin/bash

#--syllable_parser_checkpoints /tinkoff/syllable_parser_checkpoints \ 

docker run -it -v /toshiba:/toshiba -v /tinkoff:/tinkoff standy/py_asr convert_to_tfrecords \
	--feat_type spectrogram \
	--num_workers 1 \
	--num_shards 20 \
	--max_time_frames 1000 \
	--stride_factor 4 \
	--validation_set \
	/tinkoff/stt_data/_wavs_hash_flatten \
	/toshiba/stt/data/charlevel 
