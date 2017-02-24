#! /bin/bash
NV_GPU=$1 nvidia-docker run -it -v /tinkoff:/tinkoff standy/py_asr ctc_train --validate \
	--train_batch_size 90 --valid_batch_size 90 --checkpoints /tinkoff/ckpts/ctc/syllable_ctc \
	--shuffle_train --shuffle_valid --save_summaries_steps 50 --log_steps 50 --num_gpus 2 \
	--learning_rate 1e-4 --num_conv_filters 256,256,256,512,512,512 \
	--conv_filter_widths 17,17,17,17,17,17 --conv_strides 1,2,1,2,1,2 --rnn_type brnn_extended \
	--num_rec_layers 3 --rec_cell_type lstm --num_rec_hidden 512 /tinkoff/data/ctc_syllables/
