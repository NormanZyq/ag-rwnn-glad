echo 'hse_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=hse_grsna data.mode='sequence_only' model.structure_feat_size=100
echo 'dhfr_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=dhfr_grsna data.mode='sequence_only' model.structure_feat_size=100
echo 'imdb_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=imdb_grsna data.mode='sequence_only' model.structure_feat_size=100
echo 'p53_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=p53_grsna data.mode='sequence_only' model.structure_feat_size=100
echo 'reddit_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=reddit_grsna data.mode='sequence_only' model.structure_feat_size=25
echo 'aids_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=aids_grsna data.mode='sequence_only' model.structure_feat_size=50
echo 'proteins_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=proteins_grsna data.mode='sequence_only' model.structure_feat_size=75
echo 'mmp_grsna_sequence_only'
CUDA_VISIBLE_DEVICES=1 python src/multi_train.py experiment=mmp_grsna data.mode='sequence_only' model.structure_feat_size=56
