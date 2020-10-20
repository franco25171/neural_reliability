def cut_out_unit_segments(units_stimulus_info, clip_names, oracle_hashes_o, unit_trace, i, units):
                                        
    all_unit_traces = [0] * len(units_stimulus_info[0])
    all_unit_traces_norm = [0] * len(units_stimulus_info[0])
    all_oracles = [0] * len(units_stimulus_info[0])
    all_oracles_norm = [0] * len(units_stimulus_info[0])
    trial_idx = [0] * len(units_stimulus_info[0])
    condition_hash = [0] * len(units_stimulus_info[0])
    stimulus_type = [0] * len(units_stimulus_info[0])
    movie_name = [0] * len(units_stimulus_info[0])
    clip_number = [0] * len(units_stimulus_info[0])
    for clip,stim_hash in zip(clip_names,oracle_hashes_o):
        stim_start_stop_times = [(a[0],a[1],a[3]['trial_id'],a[3]['hash'],a[3]['stimulus_type'],
                                  a[3]['movie_name'],a[3]['clip_number']) for a in 
                                 units_stimulus_info[i] if a[2]['unit_id'] == units[i] and a[3]['hash'] == stim_hash]
        unit_traces = []
        unit_traces_norm = []
        min_trace_len = 10000000

        for n,(start,stop,index,condition,stype,mname,cnumber) in enumerate(stim_start_stop_times):
            unit_traces.append(unit_trace[start:stop])
            unit_traces_norm.append(normalize_signal(unit_trace[start:stop]))
            if min_trace_len > (stop-start):
                min_trace_len = stop-start
        for n in range(len(unit_traces)):
            unit_traces[n] = unit_traces[n][0:min_trace_len]
            unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]
            all_unit_traces[stim_start_stop_times[n][2]] = unit_traces[n]
            all_unit_traces_norm[stim_start_stop_times[n][2]] = unit_traces_norm[n]
            trial_idx[stim_start_stop_times[n][2]] = stim_start_stop_times[n][2] 
            condition_hash[stim_start_stop_times[n][2]] = stim_start_stop_times[n][3]
            stimulus_type[stim_start_stop_times[n][2]] = stim_start_stop_times[n][4]
            movie_name[stim_start_stop_times[n][2]] = stim_start_stop_times[n][5]
            clip_number[stim_start_stop_times[n][2]] = stim_start_stop_times[n][6]

        oracle = calculate_oracle(unit_traces)
        oracle_norm = calculate_oracle(unit_traces_norm)

        for n in range(len(unit_traces)):
            all_oracles[stim_start_stop_times[n][2]] = oracle[n]
            all_oracles_norm[stim_start_stop_times[n][2]] = oracle_norm[n]
            
    return trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_unit_traces, all_unit_traces_norm, all_oracles, all_oracles_norm