#!/bin/bash

module load anaconda
conda activate mnmt 
echo "training en to es fa fr it ko ru tr zh"

SAVE_DIR=/scratch4/danielk/tli104/mnmt/checkpoints/opus_en_x_8_baseline
rm -r $SAVE_DIR
mkdir -p $SAVE_DIR
TRAIN_DATA_DIR=/scratch4/danielk/tli104/opus-100-corpus/v1.0/supervised/data-bin
python3 train.py $TRAIN_DATA_DIR --arch transformer_wmt_en_de_large \
    --task translation_multi_simple_epoch --sampling-method "temperature" \
    --sampling-temperature 1 --encoder-langtok "tgt" \
    --langs "en,es,fa,fr,it,ko,ru,tr,zh" \
    --lang-pairs "en-es,en-fa,en-fr,en-it,en-ko,en-ru,en-tr,en-zh" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas "(0.9, 0.98)" \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 60000 \
    --no-epoch-checkpoints \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 16384 --update-freq 4 \
    --log-format simple --log-interval 100 --seed 1111 --fp16 --ddp-backend no_c10d \
    --save-dir $SAVE_DIR --max-source-positions 256 --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test 


DATA_DIR=/scratch4/danielk/tli104/opus-100-corpus/v1.0/supervised
MODEL_PATH=$SAVE_DIR/checkpoint_best.pt
echo "loaded checkpoint from $MODEL_PATH"
src=en
for tgt in es fa fr it ko ru tr zh
do
	pair="en-${tgt}"
	FSRC=${DATA_DIR}/${pair}/spm_encoded/opus.${pair}-test.${src}
	FTGT=${DATA_DIR}/${pair}/opus.${pair}-test.${tgt}
	FOUT=${MODEL_PATH}_results/test.${src}-${tgt}.${tgt}

	mkdir ${MODEL_PATH}_results
	cat $FSRC | python scripts/truncate.py | \
	python3 fairseq_cli/interactive.py ${DATA_DIR}/data-bin \
    		--task translation_multi_simple_epoch --encoder-langtok "tgt" --path $MODEL_PATH \
    		--langs "en,es,fa,fr,it,ko,ru,tr,zh"  \
    		--lang-pairs "en-es,en-fa,en-fr,en-it,en-ko,en-ru,en-tr,en-zh" \
    		--source-lang $src --target-lang $tgt --buffer-size 1024 --batch-size 100 \
    		--beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
		grep -P "^H" | cut -f 3- > $FOUT
	
	cat $FOUT | sacrebleu $FTGT --tokenize flores200 -m bleu chrf -f text --chrf-word-order 2 | grep -E "BLEU|chrF"
done


