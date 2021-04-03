train:
	python train.py --logdir=log/retina \
	  --label_file_path=data/CHASEDB/training.txt \
	  --batch_size=4 \
	  --max_iters=2000 \
	  --preprocess=True \
	  --transpose_conv=True \

train_distribute:
	python train_distribute.py --logdir=log/retina_multigpu \
	  --label_file_path=data/CHASEDB/training.txt \
	  --batch_size=8 \
	  --max_iters=1000 \
	  --preprocess=True \
	  --transpose_conv=True \

evaluate:
	python evaluate.py --logdir=log/retina \
	  --label_file_path=data/CHASEDB/test.txt \
	  --batch_size=1 \
	  --preprocess=True \
	  --transpose_conv=True \
	  --visulize=True \