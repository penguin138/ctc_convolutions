#! /bin/bash
docker run -it -v /tinkoff:/tinkoff -v /home/penguin138/ctc_convolutions:/ctc_conv -w /ctc_conv standy/py_asr build_tlg \
	/ctc_conv/arpa.lm /ctc_conv/lexicon.txt /tinkoff/data/ctc_syllables/meta.pickle --syllables
