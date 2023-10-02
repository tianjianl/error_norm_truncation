#!/bin/bash
echo "vocab size $1"
DIR=/scratch4/danielk/tli104/opus-100-corpus/v1.0/supervised
train_list=""
echo "opus data directory fixed to be ${DIR}"
for split in train dev test
do
	for lang in es fa fr it ko ru tr zh 
	do
		pair="en-${lang}"
		echo "${pair}"
		temp="${DIR}/${pair}/opus.${pair}-${split}.${lang},"
		train_list+=$temp
		temp="${DIR}/${pair}/opus.${pair}-${split}.en,"
		train_list+=$temp
	done
done
train_list=${train_list%?}
echo "train list = $train_list"
python3 scripts/spm_train.py --input=${train_list} \
	--model_type=bpe \
	--model_prefix=${DIR}/nmt_$1 \
	--vocab_size=$1 \
	--character_coverage=0.9995 \
	--input_sentence_size=1000000

for split in train dev test
do
	for lang in es fa fr it ko ru tr zh
	do
		pair="en-${lang}"
		echo "now encoding ${pair}-${split} with trained sentencepiece model"
		mkdir -p ${DIR}/${pair}/spm_encoded
		python3 scripts/spm_encode.py --model ${DIR}/nmt_$1.model --input ${DIR}/${pair}/opus.${pair}-${split}.${lang} --outputs ${DIR}/${pair}/spm_encoded/opus.${pair}-${split}.${lang}
		python3 scripts/spm_encode.py --model ${DIR}/nmt_$1.model --input ${DIR}/${pair}/opus.${pair}-${split}.en --outputs ${DIR}/${pair}/spm_encoded/opus.${pair}-${split}.en
	done
done

rm -rf ${DIR}/data-bin
mkdir -p ${DIR}/data-bin
cut -f1 ${DIR}/nmt_$1.vocab | tail -n +4 | sed "s/$/ 100/g" > ${DIR}/data-bin/dict.txt
for lang in es fa fr it ko ru tr zh
do
	pair="en-${lang}"
	echo "now preprocessing ${pair}-${split}"
	python3 fairseq_cli/preprocess.py --task "translation" --source-lang en --target-lang $lang \
    		--trainpref ${DIR}/${pair}/spm_encoded/opus.${pair}-train \
    		--validpref ${DIR}/${pair}/spm_encoded/opus.${pair}-dev \
   		--destdir ${DIR}/data-bin --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
    		--srcdict ${DIR}/data-bin/dict.txt --tgtdict ${DIR}/data-bin/dict.txt
done
