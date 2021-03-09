train:
	python train.py --logdir=log/no_preprocess \
	  --label_file_path=data/CHASEDB/training.txt \
	  --batch_size=1 \
	  --max_iters=2000 \
	  --preprocess=False \

evaluate:
	python evaluate.py --logdir=log/no_preprocess \
	  --label_file_path=data/CHASEDB/test.txt \
	  --batch_size=1 \
	  --preprocess=False \