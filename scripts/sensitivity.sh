echo 'aids_bias_strength_0'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.bias_strength=0

echo 'aids_bias_strength_0.01'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.bias_strength=0.01

echo 'aids_bias_strength_0.1'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.bias_strength=0.1

echo 'aids_bias_strength_10'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.bias_strength=10

echo 'aids_bias_strength_100'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.bias_strength=100

echo 'aids_bias_strength_9999'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.bias_strength=9999

echo 'imdb_bias_strength_0'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.bias_strength=0

echo 'imdb_bias_strength_0.01'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.bias_strength=0.01

echo 'imdb_bias_strength_0.1'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.bias_strength=0.1

echo 'imdb_bias_strength_10'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.bias_strength=10

echo 'imdb_bias_strength_100'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.bias_strength=100

echo 'imdb_bias_strength_9999'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.bias_strength=9999

echo 'p53_bias_strength_0'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.bias_strength=0

echo 'p53_bias_strength_0.01'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.bias_strength=0.01

echo 'p53_bias_strength_0.1'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.bias_strength=0.1

echo 'p53_bias_strength_10'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.bias_strength=10

echo 'p53_bias_strength_100'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.bias_strength=100

echo 'p53_bias_strength_9999'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.bias_strength=9999

echo 'proteins_bias_strength_0'
CUDA_VISIBLE_DEVICES=4 python src/multi_train.py experiment=proteins_grsna data.bias_strength=0

echo 'proteins_bias_strength_0.01'
CUDA_VISIBLE_DEVICES=4 python src/multi_train.py experiment=proteins_grsna data.bias_strength=0.01

echo 'proteins_bias_strength_0.1'
CUDA_VISIBLE_DEVICES=4 python src/multi_train.py experiment=proteins_grsna data.bias_strength=0.1

echo 'proteins_bias_strength_10'
CUDA_VISIBLE_DEVICES=4 python src/multi_train.py experiment=proteins_grsna data.bias_strength=10

echo 'proteins_bias_strength_100'
CUDA_VISIBLE_DEVICES=4 python src/multi_train.py experiment=proteins_grsna data.bias_strength=100

echo 'proteins_bias_strength_9999'
CUDA_VISIBLE_DEVICES=4 python src/multi_train.py experiment=proteins_grsna data.bias_strength=9999
