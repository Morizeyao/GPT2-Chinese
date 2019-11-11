python generate.py \
  --device 0 \
  --length 900 \
  --tokenizer_path cache/vocab_small.txt \
  --model_path model/final_model \
  --prefix "[CLS][MASK]" \
  --topp 1 \
  --temperature 1.0
