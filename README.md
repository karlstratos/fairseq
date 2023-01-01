# Setup

```
conda create --name fairseq python=3.8
pip install --editable ./
# CFLAGS="-stdlib=libc++" pip install --editable ./  # (on MacOS)
pip install tensorboardX fastBPE sacremoses
```

Toy data
```
cd examples/language_model/
bash prepare-wikitext-103.sh
cd wikitext-103/
head -100 wiki.train.tokens > wiki.toy.tokens
cd ../../..
TEXT=examples/language_model/wikitext-103 fairseq-preprocess --only-source --trainpref $TEXT/wiki.toy.tokens --validpref $TEXT/wiki.toy.tokens --testpref $TEXT/wiki.toy.tokens --destdir data-bin/wikitext-103-toy --workers 20
```

# Commands

one.cs: 156m transformer (d=512, D=2048, L=6, sharing) trained on 2 GPUs without fp16 (13g memory/GPU) --- 32 epochs, takes 2 days (172844.2 seconds)! Each epoch ~1.5 hours
```
fairseq-train --task language_modeling data-bin/wikitext-103 --save-dir /data/jl2529/fairs eq/wikitext-103/ --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode none --max-tokens 2048 --update-freq 16 --max-update 50000 --distributed-world-size 2
```
Same thing with fp16
```
fairseq-train --task language_modeling data-bin/wikitext-103 --save-dir /data/jl2529/fairs eq/wikitext-103/ --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --tokens-per-sample 512 --sample-break-mode none --max-tokens 2048 --update-freq 16 --max-update 50000 --distributed-world-size 2
```


Decoding
```
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 400  # test ppl 29.70, 69.1s (3552.90 tokens/s)
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 0  # test ppl 32.01, 18.3s (13402.89 tokens/s)
fairseq-eval-lm data-bin/wikitext-103 --path /data/jl2529/fairseq/wikitext-103/checkpoint_best.pt --batch-size 2 --tokens-per-sample 512 --context-window 511  # test ppl 29.94, 7436.0s (33.02 tokens/s) > 2 hrs!
```

# Debugging

TODO: disable train shuffle
```
fairseq-train --task language_modeling data-bin/wikitext-103-toy --save-dir /tmp/fairseq-lm-toy --arch transformer_lm --share-decoder-input-output-embed --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01  --lr 0.0005 --tokens-per-sample 10 --max-tokens 30 --max-update 10 --decoder-input-dim 4 --decoder-embed-dim 4 --decoder-output-dim 4 --decoder-ffn-embed-dim 6 --decoder-layers 1 --decoder-attention-heads 2 --log-format simple --log-interval 1 --no-save --num-workers 0
```