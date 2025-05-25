# Cora
python main_gvn_node_gcn.py  --nolabel --VE resnet50 --hop_num 2 --color_center --dataset cora --gnn_model GCN --gnnlr 0.01 --prelr 0.01 --l2 5e-4 --num_layers 1  --dropout 0.3 --num_layers_predictor 3   --hidden_channels 128 --testbs 512 --epochs 9999 --eval_steps 5  --batch_size 1024 --attn_lr 1e-3 --velr 0.01  --kill_cnt 10 --fp_dim 2048 --attention_dim 2048

# Citeseer
python main_gvn_node_gcn.py  --nolabel --VE resnet50 --hop_num 2 --color_center --dataset citeseer --gnn_model GCN --gnnlr 0.01 --prelr 0.01 --l2 5e-4 --num_layers 1  --dropout 0.3 --num_layers_predictor 3   --hidden_channels 128 --testbs 512 --epochs 9999 --eval_steps 5  --batch_size 1024 --attn_lr 1e-3 --velr 0.01  --kill_cnt 10 --fp_dim 2048 --attention_dim 2048

# Pubmed
python main_gvn_node_gcn.py  --nolabel --VE resnet50 --hop_num 2 --color_center --dataset pubmed --gnn_model GCN --gnnlr 0.01 --prelr 0.01 --l2 1e-4 --num_layers 1  --dropout 0.1 --num_layers_predictor 2   --hidden_channels 256 --testbs 512 --epochs 9999 --eval_steps 5  --batch_size 1024 --attn_lr 1e-3 --velr 1e-3  --kill_cnt 10 --fp_dim 2048 --attention_dim 2048

# Collab
python main_gvn_node_gcn_collabppaddi.py --num_layers_predictor 3 --nolabel --VE resnet50 --dropout 0. --attn_lr 0.001 --attention_dim 512 --hop_num 2 --color_center --dataset collab  --xdp 0.25 --tdp 0.05  --gnnedp 0.25  --gnnlr 0.001  --prelr 0.001 --gnndp  0.3  --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 1000  --batch_size 32768  --ln --model gcn  --testbs 131072  --maskinput --use_val_edges_as_input --res

# PPA
python main_gvn_node_gcn_collabppaddi.py --num_layers_predictor 3 --nolabel --VE resnet50 --dropout 0.3 --attn_lr 0.001  --attention_dim 512 --hop_num 1 --color_center --dataset ppa  --xdp 0.0 --tdp 0.0 --gnnedp 0.1  --gnnlr 0.001 --prelr 0.001 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 50 --kill_cnt 20 --batch_size 65536  --ln --model gcn --maskinput  --res --testbs 65536

# Citation2
python main_gvn_node_gcn_citation2.py --nolabel --VE resnet50 --velr 0.001 --attn_lr 1e-3  --attention_dim 512 --fp_dim 512 --hop_num 1 --color_center --dataset citation2  --data_name ogbl-citation2 --gnn_model GCN --hidden_channels 128 --lr 0.001 --dropout 0.3  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 32768

# DDI
python main_gvn_node_gcn_collabppaddi.py --num_layers_predictor 3 --nolabel --VE resnet50 --dropout 0.3 --attn_lr 0.001  --attention_dim 512 --hop_num 1 --color_center --dataset ddi  --xdp 0.0 --tdp 0.0 --gnnedp 0.1  --gnnlr 0.001 --prelr 0.001 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 9999 --kill_cnt 500 --batch_size 65536  --ln --model gcn --maskinput  --res --testbs 65536
