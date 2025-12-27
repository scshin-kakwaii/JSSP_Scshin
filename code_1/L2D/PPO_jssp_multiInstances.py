from mb_agg import *
from agent_utils import eval_actions, select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from Params import configs
from validation import validate

device = torch.device(configs.device)

class Memory:
    def __init__(self):
        self.adj_mb, self.fea_mb, self.candidate_mb, self.mask_mb, self.rule_fea_mb = [], [], [], [], []
        self.a_mb, self.r_mb, self.done_mb, self.logprobs = [], [], [], []
    def clear_memory(self):
        del self.adj_mb[:]; del self.fea_mb[:]; del self.candidate_mb[:]; del self.mask_mb[:]; del self.rule_fea_mb[:]
        del self.a_mb[:]; del self.r_mb[:]; del self.done_mb[:]; del self.logprobs[:]

class PPO:
    def __init__(self, lr, gamma, k_epochs, eps_clip, n_j, n_m, num_layers, neighbor_pooling_type, input_dim, hidden_dim, num_mlp_layers_feature_extract, num_mlp_layers_actor, hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic):
        self.lr = lr; self.gamma = gamma; self.eps_clip = eps_clip; self.k_epochs = k_epochs
        self.policy = ActorCritic(n_j, n_m, num_layers, False, neighbor_pooling_type, input_dim, hidden_dim, num_mlp_layers_feature_extract, num_mlp_layers_actor, hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic, device)
        self.policy_old = deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool, current_ent_coef):
        self.policy.train()
        all_rewards, all_adj, all_fea, all_cand, all_mask, all_rule_fea, all_actions, all_old_probs = [], [], [], [], [], [], [], []
        
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal: discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            all_rewards.extend(rewards)
            all_adj.extend(memories[i].adj_mb); all_fea.extend(memories[i].fea_mb)
            all_cand.extend(memories[i].candidate_mb); all_mask.extend(memories[i].mask_mb)
            all_rule_fea.extend(memories[i].rule_fea_mb); all_actions.extend(memories[i].a_mb)
            all_old_probs.extend(memories[i].logprobs)

        batch_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-5)
        
        batch_fea = torch.stack(all_fea).to(device)
        batch_cand = torch.stack(all_cand).to(device)
        batch_mask = torch.stack(all_mask).to(device)
        batch_rule_fea = torch.stack(all_rule_fea).to(device)
        batch_actions = torch.stack(all_actions).to(device).detach()
        batch_old_probs = torch.stack(all_old_probs).to(device).detach()
        
        total_steps = len(all_adj)
        indices = np.arange(total_steps)
        mini_batch_size = 64
        avg_loss = 0
        target_kl = 0.015
        
        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            epoch_kls = []
            for start in range(0, total_steps, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                
                mb_adj_list = [all_adj[i] for i in mb_idx]
                mb_adj_batch = aggr_obs(torch.stack(mb_adj_list).to(device).to_sparse(), n_tasks)
                mb_g_pool = g_pool_cal(configs.graph_pool_type, torch.Size([len(mb_idx), n_tasks*n_tasks]), n_tasks, device)
                
                pis, vals = self.policy(batch_fea[mb_idx].reshape(-1, batch_fea.size(-1)), mb_g_pool, None, mb_adj_batch, batch_cand[mb_idx], batch_mask[mb_idx], batch_rule_fea[mb_idx])
                
                logprobs, ent_loss = eval_actions(pis, batch_actions[mb_idx])
                ratios = torch.exp(logprobs - batch_old_probs[mb_idx])
                advantages = batch_rewards[mb_idx] - vals.view(-1).detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = configs.vloss_coef * self.V_loss_2(vals.squeeze(), batch_rewards[mb_idx]) + configs.ploss_coef * (-torch.min(surr1, surr2).mean()) - current_ent_coef * ent_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
                
                with torch.no_grad():
                    log_ratio = logprobs - batch_old_probs[mb_idx]
                    epoch_kls.append(((torch.exp(log_ratio) - 1) - log_ratio).mean().item())
            
            if np.mean(epoch_kls) > target_kl * 1.5: break
                
        self.policy_old.load_state_dict(self.policy.state_dict())
        return avg_loss / (self.k_epochs * (total_steps // mini_batch_size + 1))

def main():
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen
    
    envs = [SJSSP(configs.n_j, configs.n_m) for _ in range(configs.num_envs)]
    
    # Đặt DataGen cùng cấp với file PPO_jssp_multiInstances.py
    datagen_dir = Path(__file__).parent / 'DataGen'
    datagen_dir.mkdir(parents=True, exist_ok=True)
    data_path = datagen_dir / f'generatedData{configs.n_j}_{configs.n_m}_Seed{configs.np_seed_validation}.npy'
    if data_path.exists():
        vali_data = [(d[0], d[1]) for d in np.load(data_path, allow_pickle=True)]
    else:
        print("Gen new vali data...")
        vali_data = [uni_instance_gen(configs.n_j, configs.n_m, configs.low, configs.high) for _ in range(20)]

    memories = [Memory() for _ in range(configs.num_envs)]
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip, configs.n_j, configs.n_m, configs.num_layers, configs.neighbor_pooling_type, configs.input_dim, configs.hidden_dim, configs.num_mlp_layers_feature_extract, configs.num_mlp_layers_actor, configs.hidden_dim_actor, configs.num_mlp_layers_critic, configs.hidden_dim_critic)
    
    log, val_log, record = [], [], 100000
    log_file = Path(__file__).parent / f'log_{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}.txt'
    val_file = Path(__file__).parent / f'vali_{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}.txt'
    
    states = []
    for i, env in enumerate(envs):
        states.append(env.reset(options={'data': uni_instance_gen(configs.n_j, configs.n_m, configs.low, configs.high)})[0])

    for i_update in range(configs.max_updates):
        frac = 1.0 - (i_update - 1.0) / configs.max_updates
        curr_ent = max(configs.entloss_coef * frac, 0.001)
        
        # LR Decay
        curr_lr = 2.5e-5 if i_update < 4000 else (1e-5 if i_update < 8000 else 5e-6)
        for param_group in ppo.optimizer.param_groups: param_group['lr'] = curr_lr
        
        ep_rewards = [0] * configs.num_envs
        while True:
            with torch.no_grad():
                actions = []
                for i in range(configs.num_envs):
                    pi, _ = ppo.policy_old(
                        torch.from_numpy(states[i]['fea']).to(device),
                        g_pool_cal(configs.graph_pool_type, torch.Size([1, configs.n_j*configs.n_m]), configs.n_j*configs.n_m, device),
                        None,
                        torch.from_numpy(states[i]['adj']).to(device).to_sparse(),
                        torch.from_numpy(states[i]['candidate']).to(device).unsqueeze(0),
                        torch.from_numpy(states[i]['mask']).to(device).unsqueeze(0),
                        torch.from_numpy(states[i]['rule_features']).to(device).unsqueeze(0)
                    )
                    actions.append(select_action(pi, memories[i]))
            
            next_states, done_any = [], False
            for i in range(configs.num_envs):
                obs, r, d, _, _ = envs[i].step(actions[i][0].item())
                memories[i].adj_mb.append(torch.from_numpy(states[i]['adj']).to(device).to_sparse())
                memories[i].fea_mb.append(torch.from_numpy(states[i]['fea']).to(device))
                memories[i].candidate_mb.append(torch.from_numpy(states[i]['candidate']).to(device))
                memories[i].mask_mb.append(torch.from_numpy(states[i]['mask']).to(device))
                memories[i].rule_fea_mb.append(torch.from_numpy(states[i]['rule_features']).to(device))
                memories[i].a_mb.append(actions[i][0])
                memories[i].r_mb.append(r); memories[i].done_mb.append(d)
                ep_rewards[i] += r
                next_states.append(obs)
                if d: done_any = True
            
            states = next_states
            if done_any:
                states = [env.reset(options={'data': uni_instance_gen(configs.n_j, configs.n_m, configs.low, configs.high)})[0] for env in envs]
                break
        
        loss = ppo.update(memories, configs.n_j*configs.n_m, configs.graph_pool_type, curr_ent)
        for m in memories: m.clear_memory()
        
        avg_rew = sum(ep_rewards)/len(ep_rewards)
        log.append([i_update+1, avg_rew])
        with open(log_file, 'w') as f: f.write(str(log))
        
        if (i_update+1) % 10 == 0: print(f'Ep {i_update+1} | Reward: {avg_rew:.4f} | Loss: {loss:.4f} | LR: {curr_lr:.1e}')
        
        if (i_update+1) % 100 == 0:
            val_res = validate(vali_data[:20], ppo.policy).mean()
            val_log.append([i_update+1, val_res])
            with open(val_file, 'w') as f: f.write(str(val_log))
            if val_res < record:
                record = val_res
                torch.save(ppo.policy.state_dict(), Path(__file__).parent / f'{configs.n_j}_{configs.n_m}_Best.pth')
                print(f'>>> VALIDATION: {val_res:.2f} (NEW BEST!)')
            else: print(f'>>> VALIDATION: {val_res:.2f}')

if __name__ == '__main__': main()