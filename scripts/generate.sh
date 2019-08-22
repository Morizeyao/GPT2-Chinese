python generate_texts_seo.py \
  --device 0 \
  --length 900 \
  --model_path model/final_model \
  --prefix "[CLS][MASK]" \
  --topp 1 \
  --temperature 1.0
