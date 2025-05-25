# Cora
python main_gvn_link_gcn.py --id 5 --hop_num 1 --data_name cora --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 100 --kill_cnt 10 --eval_steps 1  --batch_size 256 --fslr 1e-3  --VE resnet50_tuned --velr 1e-3 --modal_fusion attn_fusion

# Citeseer
python main_gvn_link_gcn.py --id 5 --hop_num 1 --data_name citeseer  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 100 --kill_cnt 10 --eval_steps 1  --batch_size 128 --fslr 0.01  --VE resnet50_tuned --velr 0.01 --modal_fusion attn_fusion --clip_score

# Pubmed
python main_gvn_link_gcn.py --id 3 --hop_num 1 --data_name pubmed  --gnn_model GCN --lr 0.01 --dropout 0.1 --l2 0 --num_layers 1  --num_layers_predictor 2 --hidden_channels 256 --epochs 100 --kill_cnt 10 --eval_steps 1  --batch_size 256 --fslr 1e-7  --VE resnet50_tuned --velr 1e-7 --modal_fusion attn_fusion --clip_score