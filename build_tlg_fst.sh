#! /bin/bash

#docker run -it -v /tinkoff:/tinkoff -v /home/penguin138/ctc_convolutions:/ctc_conv -w /ctc_conv standy/py_asr build_tlg \
#	/ctc_conv/arpa.lm /ctc_conv/lexicon.txt /tinkoff/data/ctc_syllables/meta.pickle --syllables

#docker run -it -v /toshiba:/toshiba -v /home/penguin138/ctc_convolutions/new_tlgfst:/new_tlgfst -w /new_tlgfst standy/py_asr build_tlg \
#	/new_tlgfst/arpa.lm /new_tlgfst/original_lexicon.txt /toshiba/stt/data/syllable_level/meta.pickle --syllables

docker run -it -v /tinkoff:/tinkoff -v /home/andrew/ngram_test:/ngram_test -w /tinkoff/tlgfst \
       	standy/py_asr:eesen_decode_test build_tlg \
       	/ngram_test/arpa.lm /ngram_test/word_vocab.txt /tinkoff/ckpts/ctc/ckpt_resnet_29_new_pipeline/data_graph.pickle
