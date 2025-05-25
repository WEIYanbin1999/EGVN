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
from torch_geometric.utils import k_hop_subgraph, negative_sampling
from torch.backends import cudnn
from copy import deepcopy

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    seed_torch(worker_seed % (2**32))


class GraphVisualizer:
    def __init__(self):
        self.layout_list = ['dot', 'neato', 'circo', 'twopi', 'fdp', 'sfdp']
        
    def convert_graph_to_image(self, src_node_index, dst_node_index, edge_index, file_path,
                               store_flag='vary', layout_aug=False, nolabel = False):
        src_node_index, dst_node_index = (min(src_node_index, dst_node_index), max(src_node_index, dst_node_index))
        if store_flag == 'vary' or not os.path.exists(file_path):
            file_path = file_path.split('.png')[0]
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            
            if layout_aug:
                dot = graphviz.Graph(format='png', engine=random.choice(self.layout_list))
            else:
                dot = graphviz.Graph(format='png', engine='sfdp')

            unique_nodes = torch.unique(edge_index).cpu().numpy()

            for node in unique_nodes:
                if node not in [src_node_index, dst_node_index]:
                    if nolabel: dot.node(str(node), shape='box', label='')
                    else: dot.node(str(node), shape='box')

            for node in [src_node_index, dst_node_index]:
                if nolabel: dot.node(str(node), shape='box', color='brown', style='filled', label = '')
                else: dot.node(str(node), shape='box', color='brown', style='filled')
            # Add edges, using subgraph labels
            edge_index = edge_index.t().tolist()
            edge_list = []
            for start, end in edge_index:
                start, end = (min(start, end), max(start, end))
                if (start, end) != (src_node_index, dst_node_index) and start != end:
                    if (start, end) not in edge_list:
                        dot.edge(str(start), str(end))
                        edge_list.append((start, end))

            dot.render(filename=file_path, cleanup=True)


        
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
        # print(image)
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

    def forward(self, image, device):
        image = image.to(device)
        
        # 使用VGG16提取特征
        with torch.no_grad():
            x = self.features(image)
            x = self.avgpool(x)
        
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
        combined_features = torch.cat((node_features, image_features_expanded), dim=0)
        
        attention_weights = self.attention_layer(combined_features)
        updated_features = attention_weights * image_features_expanded
        
        # 用 g 控制残差连接和原特征的混合程度
        residual_features = self.g * node_features + (1 - self.g) * updated_features
        
        return residual_features

class FusionAttentionModule(nn.Module):
    def __init__(self, node_feature_dim, image_feature_dim, attention_dim):
        self.image_feature_dim = image_feature_dim
        super(FusionAttentionModule, self).__init__()
        self.feature_transform = nn.Linear(image_feature_dim, node_feature_dim)
        self.attention_layer = nn.Sequential(
            nn.Linear(node_feature_dim * 2, attention_dim),  # 注意这里的维度变化
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=-1)
        )
        self.f = torch.nn.Parameter(torch.ones(1))  # 可学习的混合系数
        self.g = torch.nn.Parameter(torch.ones(1))  # 可学习的混合系数

    def forward(self, src_features, dst_features, image_features):
        image_features_transformed = self.feature_transform(image_features)
        # print(image_features_transformed.shape)
        
        src_combined_features = torch.cat((src_features, image_features_transformed), dim=1)
        src_attention_weights = self.attention_layer(src_combined_features)
        src_updated_features = src_attention_weights * image_features_transformed
        residual_src_features = self.f * src_features + (1 - self.f) * src_updated_features
        
        dst_combined_features = torch.cat((dst_features, image_features_transformed), dim=1)
        dst_attention_weights = self.attention_layer(dst_combined_features)
        dst_updated_features = dst_attention_weights * image_features_transformed
        residual_dst_features = self.g * dst_features + (1 - self.g) * dst_updated_features
        return residual_src_features, residual_dst_features

class FusionConcateModule(nn.Module):
    def __init__(self, node_feature_dim, image_feature_dim, hidden_dim, output_dim, num_layers, dropout):
        self.image_feature_dim = image_feature_dim
        super(FusionConcateModule, self).__init__()
        self.feature_transform = nn.Linear(image_feature_dim, node_feature_dim)
        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(node_feature_dim * 3, output_dim))
        else:
            self.lins.append(torch.nn.Linear(node_feature_dim * 3, hidden_dim))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(torch.nn.Linear(hidden_dim, output_dim))
        self.dropout = dropout
        

    def forward(self, src_features, dst_features, image_features):
        image_features_transformed = self.feature_transform(image_features)
        x = torch.cat((src_features, dst_features), dim=1)
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
class FusionMOE(nn.Module):
    def __init__(self):
        super(FusionMOE, self).__init__()
        self.g = torch.nn.Parameter(torch.ones(1))  # 可学习的混合系数
        
    def forward(self, gnn_score, vision_score):
        final_score = self.g * gnn_score + (1 - self.g) * vision_score
        return final_score
        
def integrate_image_features(x_batch, image, feature_extractor, attention_module, device):
    # 提取图像特征
    image_features = feature_extractor(image, device)
    
    # 使用注意力机制更新节点特征
    x_batch_updated = attention_module(x_batch, image_features)
    
    return x_batch_updated


def preprocess_image(image_path):
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(242),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加一个批处理维度
    return image

def preprocess_batch_image(args, edge_list, data_name, pos_edge_flag = True, neg_edge_to_graph_index = {}, id=0):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_images = []
    hop_num_label = "" if args.hop_num == 1 else f'hop_{args.hop_num}/'
    # use_val_label = "useval/" if args.use_val_edges_as_input else ''
    use_val_label = ''
    
    if args.no_label:
        no_label_label = 'link_nolabel/' 
    elif args.relabel:
        no_label_label = 'link_relabel/'
    else:
        no_label_label = ''
    for edge in edge_list:
        src_node, dst_node = edge
        if pos_edge_flag:
            image_path = f'./dataset/{data_name}/{hop_num_label}{use_val_label}{no_label_label}{src_node}_{dst_node}.png'
        else:
            graph_index = neg_edge_to_graph_index['_'.join((str(edge[0]),str(edge[1])))]
            image_path = f'./dataset/{data_name}/{hop_num_label}{use_val_label}{no_label_label}tmp_neg_{graph_index}_{id}.png'
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        processed_images.append(image)
    # 把processed_image的第一维转化成tensor的batchsize
    batched_images = torch.stack(processed_images, dim=0)
    return batched_images


def preprocess_batch_image_vsf2sf(args, edge_list, data_name, pos_edge_flag = True, neg_edge_to_graph_index = {}, id=0):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_images = []
    hop_num_label = "" if args.hop_num == 1 else f'hop_{args.hop_num}/'
    use_val_label = ''
    store_config = "store" if pos_edge_flag else "vary"
    no_label_label = "nolabel" if args.no_label else ""

    for edge in edge_list:
        src_node, dst_node = edge
        image_path = f'./vsf2sf_{data_name}_{hop_num_label}{use_val_label}{no_label_label}{src_node}_{dst_node}.png'
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        processed_images.append(image)
    # 把processed_image的第一维转化成tensor的batchsize
    batched_images = torch.stack(processed_images, dim=0)
    return batched_images


from torch_geometric.loader import NeighborSampler

def get_subgraph(node, data, split, shuffle_flag=True, batch_size = 1, sizes=[10, 10]):
    # 使用NeighborSampler来获取以node为中心的两跳子图
    edge_index = data.edge_index if split == 'train' else data.full_edge_index
    subgraph_loader = NeighborSampler(edge_index, node_idx=torch.tensor([node], dtype=torch.long), sizes=sizes, batch_size=batch_size, 
                                      shuffle=shuffle_flag, num_workers=0, worker_init_fn=worker_init_fn)
    subgraph_data = next(iter(subgraph_loader))
    return subgraph_data

        
        
class FusionModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, h1, h2, image_features):
        # h1, h2: 链接预测节点表征，维度为 (batch_size, node_representation_size)
        # image_features: 图像特征，维度为 (batch_size, image_feature_size)
        
        # 将节点表征和图像特征拼接起来
        # print(h1.shape, h2.shape, image_features.shape)
        h = h1 * h2
        fused_representation = torch.cat((h1, h2, image_features), dim=1)
        
        # 使用多层感知机进行融合
        fused_output = torch.sigmoid(self.fc2(torch.relu(self.fc1(fused_representation))))
        
        return fused_output

import torch.nn.functional as F  
class mlp_vision_decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_vision_decoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i):
        x = x_i

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)