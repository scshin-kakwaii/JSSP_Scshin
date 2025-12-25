import torch

def aggr_obs(obs_mb, n_node):
    # obs_mb is [batch, n_nodes_each_state, fea_dim]
    # Chuyển đổi sparse tensor batch thành 1 sparse tensor lớn
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    
    # Tính toán chỉ số row/col mới cho batch lớn
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    
    # FIX: Dùng cú pháp mới để tránh Warning
    adj_batch = torch.sparse_coo_tensor(indices=idx_mb,
                                        values=vals,
                                        size=torch.Size([obs_mb.shape[0] * n_node,
                                                         obs_mb.shape[0] * n_node]),
                                        device=obs_mb.device)
    return adj_batch

def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size[0] là số lượng mẫu trong batch
    
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0] * n_nodes,),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes,),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device)
                          
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    
    # Lặp lại index cho mỗi node trong graph
    idx_0 = idx_0.repeat_interleave(n_nodes)
    
    idx_1 = torch.arange(start=0, end=n_nodes * batch_size[0],
                         device=device,
                         dtype=torch.long)
                         
    idx = torch.stack((idx_0, idx_1))
    
    # FIX: Dùng cú pháp mới
    graph_pool = torch.sparse_coo_tensor(indices=idx, 
                                         values=elem,
                                         size=torch.Size([batch_size[0],
                                                          n_nodes * batch_size[0]]),
                                         device=device)

    return graph_pool