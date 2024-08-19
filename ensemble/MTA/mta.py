import torch
import torch.nn.functional as F 
# Purpose: Computes the Gaussian kernel density estimate of data points relative to the mode.
def gaussian_kernel(mu, bandwidth, datapoints):
    # 计算数据点到模式的欧几里得距离
    dist = torch.norm(datapoints - mu,dim=-1, p=2)
    # # 高斯核函数计算密度值
    density = torch.exp(-dist**2/(2*bandwidth**2))
    return density

# Purpose: Solves the MTA alternate procedure.
def solve_mta(model, inputs, args):

    # 在不计算梯度的情况下，从模型中提取图像特征和文本特征，以及用于调整对数几率（logits）的比例因子。
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            image_features, text_features, logit_scale = model(inputs, features=True)
    logits = image_features @ text_features.t() * logit_scale 
        
    lambda_y = args.lambda_y
    lambda_q = args.lambda_q
    max_iter = 5
    temperature = 1
    
    batch_size = image_features.shape[0]
    
    # bandwidth 计算图像特征之间的成对距离，并基于这些距离确定高斯核的带宽。
    dist = torch.cdist(image_features, image_features)
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = int(0.3 * (image_features.shape[0]-1))
    selected_distances = sorted_dist[:, 1:k+1]**2  # exclude the distance to the point itself 
    mean_distance = torch.mean(selected_distances, dim=1)
    bandwidth = torch.sqrt(0.5 * mean_distance) 
    
    # Affinity matrix based on logits 基于softmax转换后的logits计算亲和度矩阵。
    affinity_matrix = (logits/temperature).softmax(1) @ (logits/temperature).softmax(1).t()
    
    # Inlierness scores initialization: uniform 初始化inlierness得分为均匀分布
    y = torch.ones(batch_size, device=image_features.device)/batch_size
    
    # Mode initialization: original image embedding ，并设置初始模式为第一个图像特征。
    mode_init = image_features[0]
    mode = mode_init
    
    convergence = False
    th = 1e-6
    iter = 0

    # MTA算法的迭代优化过程，交替更新inlierness得分和模式。
    #
    # Inlierness步骤：根据当前模式和带宽计算密度，然后更新inlierness得分y。
    # 模式步骤：根据inlierness加权的密度更新模式。
    while not convergence:
        
        ###################
        # Inlierness step #
        ###################
        
        density = gaussian_kernel(mode, bandwidth, image_features)
    
        convergence_inlierness = False
        i = 0
        while not convergence_inlierness:
            i+=1
            old_y = y
            weighted_affinity = affinity_matrix * y.unsqueeze(0)
            y = F.softmax(1/lambda_y * (density + lambda_q * torch.sum(weighted_affinity, dim=1)), dim=-1)

            if torch.norm(old_y - y)<th or i>= max_iter:
                convergence_inlierness = True
        
        #############
        # Mode step #
        #############
        
        convergence_mode = False
        i=0
        while not convergence_mode:
            i+=1
            old_mode = mode
            density = gaussian_kernel(mode, bandwidth, image_features)
            weighted_density = density *  y
            mode = torch.sum(weighted_density.unsqueeze(1)* image_features, dim=0)/torch.sum(weighted_density)
            mode /= mode.norm(p=2, dim=-1)
            
            if torch.norm(old_mode - mode)<th or i>= max_iter:
                convergence_mode = True
        
        iter +=1
        # 当连续迭代变化小于阈值th或达到最大迭代次数时，认为算法收敛。
        if iter >= max_iter:
            convergence = True
    # 计算最终输出，即模式与文本特征的乘积，然后乘以对数几率比例因子。
    output = mode.unsqueeze(0) @ text_features.t() * logit_scale
    return output
