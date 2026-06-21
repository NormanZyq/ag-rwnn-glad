import dgl
import torch
import numpy as np
import os
from src.data.gdrcm_datamodule import GDRCMDataModule
import matplotlib.pyplot as plt
import networkx as nx
import lightning as L
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

L.seed_everything(12345, workers=True)
dgl.seed(12345)

structure_importance_list = []
other_importance_list = []

# 增加几个统计数，有用指数和无用指数
valuable_index = 0
useless_index = 0

# 将explainer封装起来方便调用
def explain_graph(model,
                  explainer,
                  data_name, 
                  num_structure_feat,
                  num_hops=2,
                  walk_length=None, 
                  num_sample=None, 
                  batch_size=None, 
                  p=None, 
                  q=None, 
                  dsl=None, 
                  down_sample_rate=None,
                  start_graph_id=0,
                  end_graph_id=None,
                  show=True,
                  save_fig=False, 
                  dataset=None,
                  postfix=''):
    global structure_importance_list, other_importance_list
    structure_importance_list = []
    other_importance_list = []

    # 1. load data
    if dataset is None:
        dm = GDRCMDataModule(name=data_name,
                             walk_length=walk_length,
                             num_sample=num_sample,
                             batch_size=batch_size,
                             sampling_method='node2vec',
                             p=p,
                             q=q, 
                             dsl=dsl, 
                             down_sample_rate=down_sample_rate, 
                             data_dir='data')
        dm.setup()
        test_dataset = dm.data_test
    else:
        test_dataset = dataset
    # make sure the bound is in the range
    if end_graph_id is None:
        end_graph_id = len(test_dataset)
    else:
        end_graph_id = min(end_graph_id, len(test_dataset))
    assert start_graph_id < end_graph_id

    cal_avg_feat_mask = False
    if end_graph_id == len(test_dataset):
        cal_avg_feat_mask = True
        
    # make dirs
    # figures will be saved to logs/explainer/{data_name}/{type}
    # type contains bar, box, feat_mask, edge_mask
    if save_fig:
        types = ['feat_mask_importance', 'feat_mask_box_plot', 'subgraph']
        for type in types:
            # make dir if not exists
            os.makedirs(f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/{type}', exist_ok=True)


    # 2. load model
    # model = GRSNCounting.load_from_checkpoint(ckpt_path,
    #                                           map_location=torch.device(device))
    # model.eval()
    # 初始化GNNExplainer
    # explainer = GNNExplainer(model=model, num_hops=num_hops, lr=lr, beta1=0.8, beta2=0.2, num_epochs=200, log=True)

    feat_mask_list = []

    # 3. explain process starts
    for graph_id in range(start_graph_id, end_graph_id):
        batch = test_dataset[graph_id]
        graphs_to_explain, labels = batch[0].to(model.device), batch[1].to(model.device)
        # 获取节点特征
        node_feat = model.concat_attrs(graphs_to_explain)

        # 使用模型进行预测
        with torch.no_grad():
            h_graph = model.forward(graphs_to_explain)
            # labels_pred = model.graph_classifier(graphs_to_explain, h_graph)
            # print(f'真实标签: {labels}, 预测标签: {labels_pred}')
        
        # 解释graph
        feat_mask, edge_mask = explainer.explain_graph(graphs_to_explain, node_feat)
        if cal_avg_feat_mask:
            feat_mask_list.append(feat_mask)
        # 执行可视化
        visualize_subgraph(graphs_to_explain,
                           edge_mask,
                           show=show, 
                           save_path=f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/subgraph/subgraph_{graph_id}.png' if save_fig else None)
        visualize_feature_importance(feat_mask, 
                                     labels, 
                                     num_structure_feat=num_structure_feat,
                                     show=show, 
                                     save_path=f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/feat_mask_importance/feat_mask_importance_{graph_id}.png' if save_fig else None)
        visualize_feature_types(feat_mask, 
                                labels, 
                                num_structure_feat=num_structure_feat,
                                show=show, 
                                save_path=f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/feat_mask_box_plot/feat_mask_box_plot_{graph_id}.png' if save_fig else None)
        # lasso_feature_selection()

    # avg feat_mask
    if cal_avg_feat_mask:
        summary = f'''
        结构特征平均重要性分数: {np.mean(structure_importance_list)}
        其他特征平均重要性分数: {np.mean(other_importance_list)}
        '''
        print(summary)
        with open(f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/summary.txt', 'w') as f:
            f.write(summary)
            
        # 因为下面的调用会让那两个数组额外增加一个元素，所以summary放在前面调用
        avg_feat_mask = torch.mean(torch.stack(feat_mask_list), dim=0)
        visualize_feature_importance(avg_feat_mask, 
                                 'N/A', 
                                 num_structure_feat=num_structure_feat,
                                 show=show, 
                                 save_path=f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/feat_mask_importance/avg_feat_mask_importance.png' if save_fig else None)   
        visualize_feature_types(avg_feat_mask, 
                                'N/A',  
                                num_structure_feat=num_structure_feat,
                                show=show, 
                                save_path=f'logs/explainer/Explain_{data_name}_{num_hops}_{postfix}/feat_mask_box_plot/avg_feat_mask_box_plot.png' if save_fig else None)
        

# 可视化子图
def visualize_subgraph(graph, edge_mask, threshold=0.5, show=True, save_path=None):
    edge_mask = edge_mask.cpu().numpy()
    src, dst = graph.edges()
    g = dgl.to_networkx(graph)

    edge_weights = edge_mask > threshold
    edges_to_draw = [(src[i].item(), dst[i].item()) for i in range(len(src)) if edge_weights[i]]

    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))
    nx.draw(g, pos, edge_color='gray', node_color='red', alpha=0.5, with_labels=True)
    nx.draw_networkx_edges(g, pos, edgelist=edges_to_draw, edge_color='blue', width=2)
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.5, label='Original Edges'),
        Line2D([0], [0], color='blue', label='Important Edges')
    ]
    plt.legend(handles=legend_elements)
    
    # 添加标题显示重要边的数量
    num_important = sum(edge_mask > threshold)
    plt.title(f'Subgraph Explanation\n{num_important} important edges (threshold={threshold})')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

# 分析特征掩码
def analyze_feat_mask(feat_mask):
    # 将特征掩码转换为numpy数组并获取重要性分数
    feat_importance = feat_mask.cpu().numpy()
    
    # 计算重要性分数的统计信息
    mean_importance = np.mean(feat_importance)
    std_importance = np.std(feat_importance)
    
    # 使用z-score来识别显著重要和不重要的特征
    z_scores = (feat_importance - mean_importance) / std_importance
    
    # 找出显著重要的特征(z-score > 1)和显著不重要的特征(z-score < -1)
    significant_important = np.where(z_scores > 1)[0]
    significant_unimportant = np.where(z_scores < -1)[0]

    return significant_important, significant_unimportant


def visualize_feature_importance(feat_mask, current_label, num_structure_feat=None, show=True, save_path=None):
    # Convert feature mask to numpy array
    feat_importance = feat_mask.cpu().numpy()
    important_features, unimportant_features = analyze_feat_mask(feat_mask)
    
    # Create color mapping
    colors = ['lightgray'] * len(feat_importance)
    for idx in important_features:
        colors[idx] = 'red'
    for idx in unimportant_features:
        colors[idx] = 'blue'
        
    # Plot bar chart with a separation between structure features and other features if num_structure_feat is provided
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feat_importance)), feat_importance, color=colors, alpha=0.7)
    
    # Add vertical line to separate structure features if num_structure_feat is provided
    # 需求变更：结构特征在feat的末尾
    if num_structure_feat is not None:
        plt.axvline(x=len(feat_importance) - num_structure_feat - 0.5, color='black', linestyle='--', alpha=0.5)
        
        # 获取y轴的范围并调整文本位置
        ymin, ymax = plt.ylim()
        text_y_pos = ymax + (ymax - ymin) * 0.05  # 在最大值上方添加5%的空间
        
        plt.text((len(feat_importance) - num_structure_feat - 0.5) / 2, text_y_pos, 'Other Features', 
                horizontalalignment='center', verticalalignment='bottom')
        plt.text(len(feat_importance) - num_structure_feat/2, text_y_pos, 'Structure Features',
                horizontalalignment='center', verticalalignment='bottom')
        
        # 调整图表的上边距以容纳文本
        plt.margins(y=0.2)
        
    # Add legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant Important Features'),
        Patch(facecolor='blue', alpha=0.7, label='Significant Unimportant Features'),
        Patch(facecolor='lightgray', alpha=0.7, label='Normal Features')
    ]
    plt.legend(handles=legend_elements)
    
    # Add title and labels
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title(f'Feature Importance Visualization (label={current_label})')
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def visualize_feature_types(feat_mask, current_label, num_structure_feat=64, show=True, save_path=None):
    # 分析结构特征和其他特征的重要性，structure_feat在feat的末尾
    structure_feat_mask = feat_mask[-num_structure_feat:]
    other_feat_mask = feat_mask[:-num_structure_feat]

    # 计算平均重要性分数
    avg_structure_importance = structure_feat_mask.mean().item()
    avg_other_importance = other_feat_mask.mean().item()
    structure_importance_list.append(avg_structure_importance)
    other_importance_list.append(avg_other_importance)

    print("\n特征重要性分析:")
    print(f"结构特征平均重要性分数: {avg_structure_importance:.4f}")
    print(f"其他特征平均重要性分数: {avg_other_importance:.4f}")

    # 进行统计显著性检验
    t_stat, p_value = stats.ttest_ind(structure_feat_mask.cpu().numpy(), 
                                     other_feat_mask.cpu().numpy())

    print(f"\n统计检验结果:")
    print(f"t统计量: {t_stat:.4f}")
    print(f"p值: {p_value:.4f}")

    is_significant = False
    if p_value < 0.05:
        is_significant = True
        print("\n结论: 两类特征的重要性存在显著差异")
        if avg_structure_importance > avg_other_importance:
            print("结构特征显著更重要")
        else:
            print("其他特征显著更重要")
    else:
        print("\n结论: 两类特征的重要性没有显著差异")

    # 可视化两类特征的重要性分布
    plt.figure(figsize=(10, 6))
    plt.boxplot([structure_feat_mask.cpu().numpy(), other_feat_mask.cpu().numpy()],
                labels=['Structural Features', 'Other Features'])
    if avg_structure_importance > avg_other_importance:
        title = f'**Structural** vs Other Features (label={current_label}, p_value={p_value:.4f}, Significant={is_significant})'
    else:
        title = f'Structural vs **Other** Features (label={current_label}, p_value={p_value:.4f}, Significant={is_significant})'
    plt.title(title)
    plt.ylabel('Importance Score')
    plt.grid(True, linestyle='--', alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()




def lasso_feature_selection(X, y, alpha=0.01):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用Lasso进行特征选择
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    # 提取重要特征的索引
    important_features = np.where(lasso.coef_ != 0)[0]
    unimportant_features = np.where(lasso.coef_ == 0)[0]

    return important_features, unimportant_features