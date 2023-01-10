# Setup

```
conda create --name fairseq python=3.8
pip install --editable ./
# CFLAGS="-stdlib=libc++" pip install --editable ./  # (on MacOS)
pip install tensorboardX fastBPE sacremoses
```

# Toy
```
cd examples/language_model/
bash prepare-wikitext-103.sh
cd wikitext-103/
head -100 wiki.train.tokens > wiki.toy.tokens
cd ../../..
TEXT=examples/language_model/wikitext-103 fairseq-preprocess --only-source --trainpref $TEXT/wiki.toy.tokens --validpref $TEXT/wiki.toy.tokens --testpref $TEXT/wiki.toy.tokens --destdir data-bin/wikitext-103-toy --workers 20
```

## Training without saving

```
fairseq-train --task language_modeling data-bin/wikitext-103-toy --save-dir /tmp/fairseq-lm-toy --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01  --lr 0.0005 --tokens-per-sample 10 --max-tokens 30 --max-update 10 --decoder-input-dim 4 --decoder-embed-dim 4 --decoder-output-dim 4 --decoder-ffn-embed-dim 6 --decoder-layers 1 --decoder-attention-heads 2 --log-format simple --log-interval 1 --no-save --num-workers 0 --dict-file /tmp/vocab.txt
```

## Debugging

### Printing data without shuffling or cutting out examples
```
fairseq-train --task language_modeling data-bin/wikitext-103-toy --save-dir /tmp/fairseq-lm-toy --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01  --lr 0.0005 --tokens-per-sample 10 --max-tokens 30 --max-update 1000 --decoder-input-dim 4 --decoder-embed-dim 4 --decoder-output-dim 4 --decoder-ffn-embed-dim 6 --decoder-layers 1 --decoder-attention-heads 2 --log-format simple --log-interval 1  --num-workers 0 --dict-file /tmp/vocab.txt --no-save --max-epoch 1 --no-shuffle --required-batch-size-multiple 1 > /tmp/out.txt
```

### Eval
```
fairseq-train --task language_modeling data-bin/wikitext-103-toy --save-dir /tmp/fairseq-lm-toy --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01  --lr 0.0005 --tokens-per-sample 10 --max-tokens 30 --max-update 100 --decoder-input-dim 4 --decoder-embed-dim 4 --decoder-output-dim 4 --decoder-ffn-embed-dim 6 --decoder-layers 1 --decoder-attention-heads 2 --log-format simple --log-interval 1  --num-workers 0 --dict-file /tmp/vocab.txt
fairseq-eval-lm data-bin/wikitext-103-toy --path /tmp/fairseq-lm-toy/checkpoint_best.pt --batch-size 3  --tokens-per-sample 10  --context-window 8 --num-workers 0
```


# Commands

one.cs: 156m transformer (d=512, D=2048, L=6, sharing) trained on 2 GPUs without fp16 (13g memory/GPU) --- 32 epochs, takes 2 days (172844.2 seconds)! Each epoch ~1.5 hours
```
fairseq-train --task language_modeling data-bin/wikitext-103 --save-dir /data/jl2529/fairseq/wikitext-103/ --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode none --max-tokens 2048 --update-freq 16 --max-update 50000 --distributed-world-size 2
```
I ran the same with `--fp16` and it was like 30 minutes per epoch, so 3 times fater (e.g., < 1 day)


Decoding
```
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 0 --gen-subset valid
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 400  # test ppl 29.70, 69.1s (3552.90 tokens/s)
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 0  # test ppl 32.01, 18.3s (13402.89 tokens/s)
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 511  # test ppl 29.94, 7436.0s (33.02 tokens/s) > 2 hrs!
```
