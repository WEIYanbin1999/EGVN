import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os
import torch
import os
import matplotlib.pyplot as plt
import networkx as nx
import graphviz
import random
import numpy as np

# 固定随机性
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    seed_torch(worker_seed % (2**32))


def convert_subgraph_to_image_matplob(edge_index, file_path, store_flag):
    # 确保路径存在
    if store_flag == 'vary' or not os.path.exists(file_path):
        file_path_without_extension = file_path.split('.png')[0]
        if not os.path.exists(os.path.dirname(file_path_without_extension)):
            os.makedirs(os.path.dirname(file_path_without_extension))
        
        # 创建图形
        G = nx.Graph()

        # 添加边
        edge_index = edge_index.t().tolist()
        G.add_edges_from(edge_index)

        # 设置图形尺寸
        plt.figure(figsize=(2.33, 2.33), dpi=96) # figsize单位为英寸，dpi决定了每英寸的像素点数

        # 绘制图形
        nx.draw(G, with_labels=True, node_size=50, node_color="skyblue", font_size=10)

        # 保存图像，确保使用了足够的DPI来达到224x224的像素尺寸
        plt.savefig(file_path_without_extension + '.png', format='png', dpi=96)

        # 清理
        plt.close()

def convert_subgraph_to_image_graphviz(edge_index, file_path, store_flag, center_node=0, center_highlight =False, nolabel = False):
    if store_flag == 'vary' or not os.path.exists(file_path):
        # 确保目录存在
        file_path = file_path.split('.png')[0]
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        dot = graphviz.Graph(format='png', engine='sfdp')

        # 确定所有唯一的节点标签
        unique_nodes = torch.unique(edge_index).cpu().numpy()

        # 添加节点，使用原始的标签
        for node in unique_nodes:
            if node == center_node:
                if not nolabel:
                    dot.node(str(node), color='brown', shape='box', style='filled') if center_highlight else dot.node(str(node), shape='box')
                else: dot.node(str(node), color='brown', shape='box', style='filled', label= '') if center_highlight else dot.node(str(node), shape='box',label='')
            else: 
                dot.node(str(node), shape='box') if not nolabel else dot.node(str(node), shape='box', label = '')

        # 添加边，使用原始的节点标签，并确保不添加重复边
        added_edges = set()
        edge_index = edge_index.t().tolist()
        for start, end in edge_index:
            edge = (min(start, end), max(start, end))
            if edge not in added_edges:
                dot.edge(str(start), str(end))
                added_edges.add(edge)

        # 注意：这里的`file_path`不应包括`.png`扩展名，`graphviz`会自动添加
        dot.render(filename=file_path, cleanup=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
class Resnet50VE(nn.Module):
    def __init__(self):
        super(Resnet50VE, self).__init__()
        # 使用预训练的ResNet50模型提取特征
        self.resnet50 = resnet50(pretrained=True)
        # 移除最后的全连接层，以获取特征向量
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()  # 设置为评估模式

    def forward(self, image, device):
        image = image.to(device)
        
        # 使用ResNet50提取特征
        with torch.no_grad():
            features = self.resnet50(image)
        
        # 去掉不必要的维度
        features = features.squeeze()
        return features
    
class Resnet50VE_Tuned(nn.Module):
    def __init__(self):
        super(Resnet50VE_Tuned, self).__init__()
        # 使用预训练的ResNet50模型提取特征
        self.resnet50 = resnet50(pretrained=True)
        # 移除最后的全连接层，以获取特征向量
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()  # 设置为评估模式

    def forward(self, image, device):
        image = image.to(device)
        
        # 使用ResNet50提取特征
        features = self.resnet50(image)
        
        # 去掉不必要的维度
        features = features.squeeze()
        return features

from timm import create_model
    
from torchvision.models import vgg16

class VGG16FE(nn.Module):
    def __init__(self):
        super(VGG16FE, self).__init__()
        # 使用预训练的VGG16模型提取特征
        self.vgg16 = vgg16(pretrained=True)
        # 移除最后的分类层（全连接层），以获取特征向量
        self.features = nn.Sequential(*list(self.vgg16.features.children()))
        self.avgpool = self.vgg16.avgpool
        self.vgg16.eval()  # 设置为评估模式

    def forward(self, image, device="cpu"):
        image = image.to(device)
        
        # 使用VGG16提取特征
        with torch.no_grad():
            x = self.features(image)
            x = self.avgpool(x)
            x = torch.flatten(x, 1) 
        # 去掉不必要的维度
        features = torch.flatten(x, 1)
        return features
    
class VGG16FE_Tuned(nn.Module):
    def __init__(self):
        super(VGG16FE_Tuned, self).__init__()
        # 使用预训练的VGG16模型提取特征
        self.vgg16 = vgg16(pretrained=True)
        # 移除最后的分类层（全连接层），以获取特征向量
        self.features = nn.Sequential(*list(self.vgg16.features.children()))
        self.avgpool = self.vgg16.avgpool
        self.vgg16.eval()  # 设置为评估模式

    def forward(self, image, device):
        image = image.to(device)
        
        # 使用VGG16提取特征
        with torch.no_grad():
            x = self.features(image)
            x = self.avgpool(x)
            x = torch.flatten(x, 1) 
        # 去掉不必要的维度
        features = torch.flatten(x, 1)
        return features
    
class ViTFE(nn.Module):
    def __init__(self):
        super(ViTFE, self).__init__()
        # 使用预训练的Vision Transformer (ViT) 模型提取特征
        self.vit = create_model('vit_base_patch16_224', pretrained=True)
        # 移除分类头，以获取特征向量
        self.vit.head = nn.Identity()  # 将分类头替换成Identity层，保留特征向量
        self.vit.eval()  # 设置为评估模式

    def forward(self, image, device):
        image = image.to(device)
        
        # 使用ViT提取特征
        with torch.no_grad():
            features = self.vit(image)
        
        # 返回特征向量
        return features
    
    
class ViTFE_Tuned(nn.Module):
    def __init__(self):
        super(ViTFE_Tuned, self).__init__()
        # 使用预训练的Vision Transformer (ViT) 模型提取特征
        self.vit = create_model('vit_base_patch16_224', pretrained=True)
        # 移除分类头，以获取特征向量
        self.vit.head = nn.Identity()  # 将分类头替换成Identity层，保留特征向量

    def forward(self, image, device):
        image = image.to(device)
        features = self.vit(image)
        
        # 返回特征向量
        return features
class AttentionModule(nn.Module):
    def __init__(self, node_feature_dim, image_feature_dim, attention_dim):
        self.image_feature_dim = image_feature_dim
        super(AttentionModule, self).__init__()
        self.feature_transform = nn.Linear(image_feature_dim, node_feature_dim)
        self.attention_layer = nn.Sequential(
            nn.Linear(node_feature_dim * 2, attention_dim),  # 注意这里的维度变化
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.g = torch.nn.Parameter(torch.ones(1))  # 可学习的混合系数

    def forward(self, node_features, image_features):
        image_features_transformed = self.feature_transform(image_features)
        image_features_expanded = image_features_transformed.expand_as(node_features)
        combined_features = torch.cat((node_features, image_features_expanded), dim=-1)
        # print(combined_features.shape)
        attention_weights = self.attention_layer(combined_features)
        updated_features = attention_weights * image_features_expanded
        
        # 用 g 控制残差连接和原特征的混合程度
        residual_features = self.g * node_features + (1 - self.g) * updated_features
        
        return residual_features


class LinearProjector(nn.Module):
    def __init__(self, image_feature_dim, projected_dim):
        super(LinearProjector, self).__init__()
        self.feature_projector = nn.Linear(image_feature_dim, projected_dim)
    def forward(self, image_features):
        transformed_features = self.feature_projector(image_features)
        return transformed_features

class AttentionModule2(nn.Module):
    def __init__(self, node_feature_dim, image_feature_dim, attention_dim, fusion_dim):
        self.image_feature_dim = image_feature_dim
        super(AttentionModule2, self).__init__()
        self.feature_transform = nn.Linear(image_feature_dim, node_feature_dim)
        self.attention_layer = nn.Sequential(
            nn.Linear(node_feature_dim * 2, attention_dim),  # 注意这里的维度变化
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.g = torch.nn.Parameter(torch.ones(1))  # 可学习的混合系数
        self.feature_transform2 = nn.Linear(node_feature_dim, fusion_dim)

    def forward(self, node_features, image_features):
        image_features_transformed = self.feature_transform(image_features)
        image_features_expanded = image_features_transformed.expand_as(node_features)
        combined_features = torch.cat((node_features, image_features_expanded), dim=-1)
        
        attention_weights = self.attention_layer(combined_features)
        updated_features = attention_weights * image_features_expanded
        
        # 用 g 控制残差连接和原特征的混合程度
        residual_features = self.g * node_features + (1 - self.g) * updated_features
        residual_features = self.feature_transform2(residual_features)
        residual_features = torch.cat((node_features, residual_features), dim=-1)
        
        return residual_features
    
    
def integrate_image_features(x_batch, image, feature_extractor, attention_module, device):
    # 提取图像特征
    image_features = feature_extractor(image, device)
    
    # 使用注意力机制更新节点特征
    x_batch_updated = attention_module(x_batch, image_features)
    
    return x_batch_updated


def preprocess_image(image_path):
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加一个批处理维度
    return image

from torch_geometric.loader import NeighborSampler

def get_subgraph(node, data, split, shuffle_flag=True, batch_size = 1, sizes=[10, 10]):
    # 使用NeighborSampler来获取以node为中心的两跳子图
    edge_index = data.edge_index if split == 'train' else data.full_edge_index
    subgraph_loader = NeighborSampler(edge_index, node_idx=torch.tensor([node], dtype=torch.long), sizes=sizes, batch_size=batch_size, 
                                      shuffle=shuffle_flag, num_workers=0, worker_init_fn=worker_init_fn)
    subgraph_data = next(iter(subgraph_loader))
    return subgraph_data

from tqdm import tqdm

def generate_subgraphs(dataname, data,  hop_num = 2, feat = 5, visualizer='graphviz', relabel = False, nolabel = False, use_val= False, color_center = False, tmp_image= False):
    label_flag = ''
    if relabel: label_flag = 'relabel_'
    if nolabel: label_flag = 'nolabel_'
    unvisual_start_node = 0
    val_flag = '_use_val' if use_val else ""
    # val_flag = ""
    if tmp_image:
        # 加载预训练的ResNet模型
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        resnet = nn.Sequential(*modules)
        resnet.eval()
        hop_label = 'mulhop' if hop_num == 2 else 'trihop'
        # if hop_num == 1: hop_label = 'one_hop'
        output_file = f'dataset/{dataname}/{label_flag}_VSF_{hop_label}{val_flag}.pt' 
        num_nodes = data.num_nodes
        feature_dim = 2048  # ResNet-50的特征向量维度
        
        if os.path.exists(output_file):
            all_features = torch.load(output_file)
        else:
            all_features = torch.zeros((num_nodes, feature_dim))
            
        zero_vector_indices = (all_features.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if len(zero_vector_indices) > 0: unvisual_start_node = zero_vector_indices[0].item()
  
    if hop_num == 1: 
        sizes = [feat,]
    elif hop_num == 2: 
        sizes = [feat, feat]
    elif hop_num == 3:
        sizes = [feat, feat, feat]

    
    print('unvisual_start_node:', unvisual_start_node)
    if tmp_image and len(zero_vector_indices) == 0: 
        print(all_features)
        return
    for node in tqdm(range(unvisual_start_node, data.num_nodes)):
        
        bsz, node_mapping, adjs = get_subgraph(node, data, split='train', sizes=sizes)
        subgraph_adj = adjs[0].edge_index if hop_num >= 2 else adjs.edge_index
        if not relabel:
            graph_edge_index = node_mapping[subgraph_adj]
        else:
            graph_edge_index = subgraph_adj
        if tmp_image:
            image_path = f'dataset/{dataname}/{label_flag}_tmp{val_flag}.png'
            convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'vary', center_node=node, center_highlight= color_center, nolabel = nolabel)
        else:
            image_path =  f'dataset/{dataname}/{label_flag}image/subgraph_hop_{hop_num}_node_{node}_{val_flag}.png'
            convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'store', center_node=node, center_highlight= color_center, nolabel = nolabel)
    
        # 图像是临时的 代表需要存储的是VE后的视觉向量
        if tmp_image:
            processed_image = preprocess_image(image_path)
            with torch.no_grad():  features = resnet(processed_image).squeeze()
            all_features[node] = features     
            if node % 1000 == 0 and node !=0 or node == data.num_nodes -1: torch.save(all_features, output_file)
    return

def generate_subgraphs_specify_VE(args, dataname, data,  hop_num = 2, feat = 5, visualizer='graphviz', relabel = False, nolabel = False, use_val= False, color_center = False, tmp_image= False):
    image_feature_dim_dict = {'resnet50':2048, 'resnet50_tuned':2048, 'vgg16':512*7*7, 'vit':768}
    label_flag = ''
    if relabel: label_flag = 'relabel_'
    if nolabel: label_flag = 'nolabel_'
    unvisual_start_node = 0
    val_flag = '_use_val' if use_val else ""
    ve_flag = "" if args.VE == 'resnet50' else args.VE
    # val_flag = ""
    if tmp_image:
        if args.VE == 'resnet50':
            ve = resnet50(pretrained=True)
            modules = list(ve.children())[:-1]
            ve = nn.Sequential(*modules)
            ve.eval()
        elif args.VE == 'vgg16':
            # 加载预训练的VGG16模型
            ve = VGG16FE()
        elif args.VE == 'vit':
            ve = create_model('vit_base_patch16_224', pretrained=True)  # 加载预训练的ViT模型
            ve.head = nn.Identity()  # 将分类头替换成Identity层，保留特征向量
            ve.eval()  # 设置为评估模式
        hop_label = 'mulhop' if hop_num == 2 else 'trihop'
        # if hop_num == 1: hop_label = 'one_hop'
        output_file = f'dataset/{dataname}/{label_flag}_VSF_{hop_label}{ve_flag}.pt' 
        num_nodes = data.num_nodes
        feature_dim = image_feature_dim_dict[args.VE]  # ResNet-50的特征向量维度
        
        if os.path.exists(output_file):
            all_features = torch.load(output_file)
        else:
            all_features = torch.zeros((num_nodes, feature_dim))
            
        zero_vector_indices = (all_features.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if len(zero_vector_indices) > 0: unvisual_start_node = zero_vector_indices[0].item()
  
    if hop_num == 1: 
        sizes = [feat,]
    elif hop_num == 2: 
        sizes = [feat, feat]
    elif hop_num == 3:
        sizes = [feat, feat, feat]

    
    print('unvisual_start_node:', unvisual_start_node)
    if tmp_image and len(zero_vector_indices) == 0: 
        print(all_features)
        return
    for node in tqdm(range(unvisual_start_node, data.num_nodes)):
        
        bsz, node_mapping, adjs = get_subgraph(node, data, split='train', sizes=sizes)
        subgraph_adj = adjs[0].edge_index if hop_num >= 2 else adjs.edge_index
        if not relabel:
            graph_edge_index = node_mapping[subgraph_adj]
        else:
            graph_edge_index = subgraph_adj
        if tmp_image:
            image_path = f'dataset/{dataname}/{label_flag}_tmp{val_flag}.png'
            convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'vary', center_node=node, center_highlight= color_center, nolabel = nolabel)
        else:
            image_path =  f'dataset/{dataname}/{label_flag}image/subgraph_hop_{hop_num}_node_{node}_{val_flag}.png'
            convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'store', center_node=node, center_highlight= color_center, nolabel = nolabel)
    
        # 图像是临时的 代表需要存储的是VE后的视觉向量
        if tmp_image:
            processed_image = preprocess_image(image_path)
            with torch.no_grad():  
                if args.VE == 'vgg16': features = ve(processed_image).squeeze()
                else: features = ve(processed_image).squeeze()
            all_features[node] = features     
            if node % 1000 == 0 and node !=0 or node == data.num_nodes -1: torch.save(all_features, output_file)
    return

import torch.multiprocessing as mp

def generate_and_process_subgraph_cpu(node, data, dataname, hop_num, feat, visualizer, relabel, nolabel, use_val, color_center, tmp_image, all_features, resnet, sizes, output_file, worker_id):
    label_flag = ''
    if relabel: label_flag = 'relabel_'
    if nolabel: label_flag = 'nolabel_'
    val_flag = '_use_val' if use_val else ""

    bsz, node_mapping, adjs = get_subgraph(node, data, split='train', sizes=sizes)
    subgraph_adj = adjs[0].edge_index if hop_num == 2 else adjs.edge_index
    if not relabel:
        graph_edge_index = node_mapping[subgraph_adj]
    else:
        graph_edge_index = subgraph_adj

    if tmp_image:
        image_path = f'dataset/{dataname}/{label_flag}_tmp_worker_{worker_id}.png'
        convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'vary', center_node=node, center_highlight=color_center, nolabel=nolabel)
        processed_image = preprocess_image(image_path)
        with torch.no_grad():
            features = resnet(processed_image.cpu()).squeeze()
        all_features[node] = features.cpu()
        if node % 1000 == 0 and node != 0:
            torch.save(all_features, output_file)
    else:
        image_path = f'dataset/{dataname}/{label_flag}image/subgraph_hop_{hop_num}_node_{node}_{val_flag}.png'
        convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'store', center_node=node, center_highlight=color_center, nolabel=nolabel)

def parallel_generate_subgraphs_cpu(dataname, data, hop_num=2, feat=5, visualizer='graphviz', relabel=False, nolabel=False, use_val=False, color_center=False, tmp_image=False, num_workers=4):
    label_flag = ''
    if relabel: label_flag = 'relabel_'
    if nolabel: label_flag = 'nolabel_'
    unvisual_start_node = 0

    if tmp_image:
        # 加载预训练的ResNet模型
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        resnet = nn.Sequential(*modules)
        resnet.eval()
        output_file = f'dataset/{dataname}/{label_flag}_VSF.pt'
        num_nodes = data.num_nodes
        feature_dim = 2048  # ResNet-50的特征向量维度

        if os.path.exists(output_file):
            all_features = torch.load(output_file)
        else:
            all_features = torch.zeros((num_nodes, feature_dim))

        zero_vector_indices = (all_features.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        if len(zero_vector_indices) > 0:
            unvisual_start_node = zero_vector_indices[0].item()
    else:
        all_features = None
        resnet = None
        output_file = None

    if hop_num == 1:
        sizes = [feat,]
    elif hop_num == 2:
        sizes = [feat, feat]
    
    val_flag = '_use_val' if use_val else ""

    if len(zero_vector_indices) == 0:
        print(all_features)
        return

    # 设置启动方法为spawn
    mp.set_start_method('spawn', force=True)

    # 创建多进程池
    pool = mp.Pool(processes=num_workers)
    jobs = []

    for node in range(unvisual_start_node, data.num_nodes):
        job = pool.apply_async(generate_and_process_subgraph_cpu, args=(node, data, dataname, hop_num, feat, visualizer, relabel, nolabel, use_val, color_center, tmp_image, all_features, resnet, sizes, output_file, node % num_workers))
        jobs.append(job)

    # 等待所有进程完成
    for job in tqdm(jobs):
        job.get()

    pool.close()
    pool.join()

    if tmp_image:
        torch.save(all_features, output_file)

    return




from torch_geometric.utils import degree

def generate_one_hop_node_subgraph(dataname, data, visualizer='graphviz', relabel = True, nolabel = True, use_val= False, color_center = True):
       
    # 计算每个节点的度数
    deg_function = degree
    # print(data.edge_index)
    deg = deg_function(data.edge_index[0], data.num_nodes)


    
    # 获取图中最大节点度数
    max_deg = int(deg.max().item())
    min_deg = int(deg.min().item())
    print(f'min_deg:{min_deg}, max_deg:{max_deg}')
    
    label_flag = ''
    if relabel: label_flag = 'relabel_'
    if nolabel: label_flag = 'nolabel_'
    
    for degree_ in tqdm(range(min_deg,max_deg+1)):
        edges_from_center = torch.zeros(degree_, dtype=torch.long)
        edges_to_center = torch.arange(1, degree_ + 1)

        graph_edge_index = torch.stack([edges_from_center, edges_to_center], dim=0)   
        image_path = f'dataset/{dataname}/{label_flag}_deg_{degree_}.png'
        convert_subgraph_to_image_graphviz(graph_edge_index, image_path, 'store', center_node=0, center_highlight= color_center, nolabel = nolabel)

    unvisual_start_node = 0
    
    # 加载预训练的ResNet模型
    resnet = resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    resnet.eval()
    output_file = f'dataset/{dataname}/{label_flag}_VSF_onehop.pt' 
    degree_VSF_output_file = f'dataset/{dataname}/{label_flag}_degree_VSF.pt' 
    num_nodes = data.num_nodes
    feature_dim = 2048  # ResNet-50的特征向量维度
    
    
    cached_VSF_by_degree = torch.zeros((max_deg+1, feature_dim))
    if os.path.exists(degree_VSF_output_file):
        cached_VSF_by_degree = torch.load(degree_VSF_output_file)
    else:
        for degree_ in tqdm(range(min_deg,max_deg+1)):
            image_path = f'dataset/{dataname}/{label_flag}_deg_{degree_}.png'
            processed_image = preprocess_image(image_path)
            with torch.no_grad():  features = resnet(processed_image).squeeze()
            cached_VSF_by_degree[degree_] = features
    torch.save(cached_VSF_by_degree, degree_VSF_output_file)
    
    
    # if dataname in ['citation2']: return
    if os.path.exists(output_file):
        all_features = torch.load(output_file)
    else:
        all_features = torch.zeros((num_nodes, feature_dim))
        
    zero_vector_indices = (all_features.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if len(zero_vector_indices) > 0: unvisual_start_node = zero_vector_indices[0].item()

    
    print('unvisual_start_node:', unvisual_start_node)
    if len(zero_vector_indices) == 0: 
        print(all_features)
        return
    
    for node in tqdm(range(unvisual_start_node, data.num_nodes)):
        degree_ = int(deg[node])
        all_features[node] = cached_VSF_by_degree[degree_]
        if node % 100000 == 0 and node !=0 or node == data.num_nodes -1: torch.save(all_features, output_file)
    return