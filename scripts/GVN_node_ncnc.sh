# Cora
python main_gvn_node_ncnc.py --nolabel --VE resnet50 --hop_num 2 --color_center --dataset cora  --gnnlr 1e-3 --prelr 1e-3 --l2 1e-3  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 1 --hiddim 128 --testbs 512 --epochs 9999 --eval_steps 5  --batch_size 1024     --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4    --probscale 4.3 --proboffset 2.8 --alpha 1.0    --ln --lnnn --predictor incn1cn1 --model puregcn   --maskinput  --jk  --use_xlin  --tailact --attn_lr 1e-4 --velr 1e-4 --kill_cnt 5 --fp_dim 2048 --attention_dim 2048

# Citeseer
python main_gvn_node_ncnc.py --nolabel --VE resnet50 --hop_num 2 --color_center --dataset citeseer  --predictor incn1cn1   --gnnlr 1e-3 --prelr 1e-3 --l2 1e-4   --predp 0.5 --gnndp 0.5  --mplayers 1 --nnlayers 2   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0  --probscale 6.5 --proboffset 4.4 --alpha 0.4 --ln --lnnn   --model puregcn  --testbs 512  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin --attn_lr 1e-4 --velr 1e-4 --fp_dim 2048

# Pubmed
python main_gvn_node_ncnc.py --nolabel --VE resnet50 --hop_num 2 --color_center --dataset pubmed    --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 300 --eval_steps 5 --kill_cnt 100 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --attn_lr 1e-7 --velr 1e-7 --fp_dim 128 --attention_dim 128

# Collab
python main_gvn_node_ncnc_collabppaddi.py --nolabel --VE resnet50 --attn_lr 0.001 --attention_dim 512 --hop_num 2 --color_center --predictor incn1cn1 --dataset collab  --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0  --gnnlr 0.001  --prelr 0.001 --predp 0.3 --gnndp  0.3  --probscale 2.5 --proboffset 6.0 --alpha 1.05   --mplayers 3  --hiddim 256  --epochs 9999 --kill_cnt 1000  --batch_size 32768  --ln --lnnn  --model gcn  --testbs 131072  --maskinput --use_val_edges_as_input   --res  --use_xlin  --tailact

# PPA
python main_gvn_node_ncnc_collabppaddi.py --nolabel --VE resnet50 --attn_lr 1e-5  --attention_dim 512 --hop_num 1 --color_center  --predictor incn1cn1 --dataset ppa  --xdp 0.0 --tdp 0.0 --gnnedp 0.1 --preedp 0.0  --gnnlr 0.001 --prelr 0.001 --predp 0 --gnndp 0 --mplayers 3 --hiddim 256 --epochs 50 --kill_cnt 20 --batch_size 65536  --ln --lnnn --model gcn --maskinput  --tailact  --res  --testbs 65536 --proboffset 8.5 --probscale 4.0 --pt 0.1 --alpha 0.9 --splitsize 131072

# Citation2
python main_gvn_node_ncnc_citation2.py --nolabel --VE resnet50 --cat --attn_lr 0.001  --attention_dim 32 --hop_num 1 --color_center --predictor incn1cn1 --dataset citation2  --xdp 0.0 --tdp 0.3 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.001 --prelr 0.001  --predp 0.3 --gnndp 0.3  --mplayers 3 --hiddim 128 --epochs 30 --kill_cnt 1000 --batch_size 16384 --ln --lnnn  --model puregcn --res --testbs 65536 --use_xlin --tailact --proboffset 4.7 --probscale 7.0 --pt 0.3 --trndeg 128 --tstdeg 128

# DDI
python main_gvn_node_ncnc_collabppaddi.py --nolabel --VE resnet50 --attn_lr 0.001  --attention_dim 512 --hop_num 1 --color_center  --predictor incn1cn1 --dataset ddi  --xdp 0.05 --tdp 0.0 --gnnedp 0.0 --preedp 0.0  --gnnlr 0.01 --prelr 0.01 --predp 0.3 --gnndp 0.3 --mplayers 1 --hiddim 256 --epochs 9999 --kill_cnt 200  --batch_size 66356   --ln --lnnn  --model puresum  --proboffset 3 --probscale 10 --pt 0.1 --alpha 0.5 --testbs 24576 --splitsize 262144  --use_xlin  --twolayerlin  --res  --maskinput 