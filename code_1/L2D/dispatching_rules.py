from enum import Enum
import numpy as np

class Rules(Enum):
    SPT = 0   
    LPT = 1   
    STPT = 2  
    LTPT = 3  
    LOR = 4   
    MOR = 5   
    LQNO = 6  
    MQNO = 7  

def get_job_and_op_indices(op_id, env):
    return op_id // env.number_of_machines, op_id % env.number_of_machines

def apply_dispatching_rule(rule, candidates, env):
    if len(candidates) == 0:
        raise ValueError("Empty candidates.")
    
    if len(candidates) == 1:
        return candidates[0]

    # --- OPTIMIZATION: Truy cập trực tiếp mảng phẳng (Flattened) để nhanh hơn ---
    candidate_durations = env.dur.flatten()[candidates]

    if rule == Rules.SPT:
        return candidates[np.argmin(candidate_durations)]

    if rule == Rules.LPT:
        return candidates[np.argmax(candidate_durations)]

    if rule == Rules.STPT:
        # Sử dụng Cached Data đã tính ở Env.reset()
        # env.job_total_times là mảng (n_j,)
        # candidates là mảng op_ids. Cần map về job_ids.
        job_indices = candidates // env.number_of_machines
        job_times = env.job_total_times[job_indices]
        return candidates[np.argmin(job_times)]

    if rule == Rules.LTPT:
        job_indices = candidates // env.number_of_machines
        job_times = env.job_total_times[job_indices]
        return candidates[np.argmax(job_times)]

    if rule == Rules.LOR:
        # Tính nhanh số operation còn lại
        # op_idx = op_id % n_m
        # remaining = n_m - op_idx
        op_indices = candidates % env.number_of_machines
        ops_remaining = env.number_of_machines - op_indices
        return candidates[np.argmin(ops_remaining)]

    if rule == Rules.MOR:
        op_indices = candidates % env.number_of_machines
        ops_remaining = env.number_of_machines - op_indices
        return candidates[np.argmax(ops_remaining)]

    if rule == Rules.LQNO or rule == Rules.MQNO:
        # Rule này động, buộc phải tính toán, nhưng có thể tối ưu nhẹ
        machine_queues = np.bincount(env.m.flatten()[env.omega[~env.mask]], minlength=env.number_of_machines+1)
        
        scores = []
        for op_id in candidates:
            job_idx, op_idx = get_job_and_op_indices(op_id, env)
            if op_idx == env.number_of_machines - 1:
                scores.append(0)
            else:
                next_m = env.m[job_idx, op_idx + 1]
                scores.append(machine_queues[next_m])
        
        if rule == Rules.LQNO:
            return candidates[np.argmin(scores)]
        else: 
            return candidates[np.argmax(scores)]

    raise NotImplementedError(f"Rule {rule} not implemented.")