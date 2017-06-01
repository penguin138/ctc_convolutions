#! /bin/bash

#--syllable_parser_checkpoints /tinkoff/syllable_parser_checkpoints \ 

NV_GPU=0 nvidia-docker run -it -v /toshiba:/toshiba -v /tinkoff:/tinkoff standy/py_asr convert \
	tinkoff_internal \
        /tinkoff/stt_data/_wavs_hash_flatten_new \
        /tinkoff/stt_data/transcriptions.json \
        /toshiba/stt/data/new_syllable_tf_records_with_old_feats \
        --count_dict /tinkoff/stt_data/1grams-3.txt \
	--audio_lib scipy \
	--quantize False \
	--feat_type spectrogram \
	--num_feats 101 \
	--sgram_value_scaling density \
	--sgram_freq_scaling hz \
	--sgram_epsilon 1 \
	--num_shards 20 \
	--max_time_frames 3000 \
	--stride_factor 4 \
	--validation_set \
	--syllable_parser_checkpoints /tinkoff/syllable_parser_checkpoints

#convert_to_tfrecords tinkoff_internal /tinkoff/stt_data/_wavs_hash_flatten_new /tinkoff/stt_data/transcriptions.json <tfrecords_output_dir> --count_dict /tinkoff/stt_data/1grams-3.txt --audio_lib scipy --quantize False --feat_type spectrogram --num_feats 101 --sgram_value_scaling density --sgram_freq_scaling hz --sgram_epsilon 1 --num_shards 20 --max_time_frames 3000 --stride_factor 4 --validation_set
