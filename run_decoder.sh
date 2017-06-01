#! /bin/bash
docker run -it -v $PWD:/decode -v /tinkoff:/tinkoff -w /decode standy/py_asr eesen_decode_calc_wer \
	--acoustic-scale 1.5 \
	--max-active 20000 \
	--beam 20.0 \
	/decode/TLG.fst /tinkoff/data/ctc_syllables /tinkoff/ckpts/ctc/syllable_ctc_new_data_flow /decode/words.txt 500
