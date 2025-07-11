OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network sphere20 --classifier MCP
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network sphere36 --classifier MCP
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv1 --classifier MCP
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv2 --classifier MCP --batch-size 256
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv3_small --classifier MCP
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 main.py --root data/train/ms1m_112x112 --database MS1M --network mobilenetv3_large --classifier MCP --batch-size 256
