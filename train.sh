OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network sphere20
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network sphere36
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network sphere64
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network sphere20 --classifier ARC
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network sphere36 --classifier ARC
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py --root data/train/ms1m_112x112 --database MS1M --network sphere64 --classifier ARC