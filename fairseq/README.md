# Machine Translation


## Getting Started 
To download and extract the data for all the languages:
```
wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz

tar -zxvf opus-100-corpus-v1.0.tar.gz
```

To preprocess the data for the eight languages (learn bpe model, tokenize data and preprocess using fairseq)
```
cd fairseq 
bash opus_8_scripts/preprocess_opus_8.sh <vocab_size>
```

## Training 
To train a baseline model from English to the eight languages (En-X) and do inference on the test set
```
bash opus_8_scripts/train_opus_8_baseline.sh
```

To use ENT-Fraction
```
bash opus_8_scripts/train_opus_8_ent_fraction.sh <start truncate iteration> <prune fraction>
```

To use ENT-Threshold
```
bash opus_8_scripts/train_opus_8_ent_threshold.sh <start truncate iteration> <prune threshold>
```
