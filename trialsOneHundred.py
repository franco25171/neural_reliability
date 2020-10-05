import datajoint as dj
import numpy as np
from pipeline import pupil, meso, treadmill
from stimulus import stimulus
from stimulus.utils import get_stimulus_info
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from stimline import tune
from sklearn.linear_model import LinearRegression
import scipy as spy
from scipy.signal import find_peaks
import math
import scipy
from scipy.signal import savgol_filter
from scipy.signal import firwin

schema = dj.schema('franco_100_trials', locals())

@schema 
class Animal(dj.Manual):
    definition = """
    animal_id : int # animal id assigned by the lab 
    ---
   
    """
    
@schema 
class Session(dj.Manual):
    definition = """
    # Experiment Session
    -> Animal
    session: smallint # session number for the animal
    ---
 
    """
    
@schema 
class ScanIdx(dj.Manual):
    definition = """
    # Two-photon imaging scan
    -> Session
    scan_idx: smallint # scan number within the session
    ---
 
    """
    
    
@schema
class PupilFilterMethod(dj.Lookup):
    definition = """
    # Variants in unit traces
    pupil_filter_method               : tinyint              # method index
    ---
    description              : varchar(250)         # description of method
    """

    contents = [{'pupil_filter_method':1, 'description':"raw pupil trace"},
               {'pupil_filter_method':2, 'description':"Butterworth low pass filtered pupil trace"},
               {'pupil_filter_method':3, 'description':"Band pass filtered pupil trace using Hamming window"}]
    
    
@schema
class UnitTraceMethod(dj.Lookup):
    definition = """
    # Variants in unit traces
    unit_trace_method               : tinyint              # method index
    ---
    description              : varchar(250)         # description of method
    """

    contents = [{'unit_trace_method':1, 'description':"raw fluorescence"},
               {'unit_trace_method':2, 'description':"deconvolved fluorescence trace"},
               {'unit_trace_method':3, 'description':"deconvolved smoothed fluorescence trace using a sgolay filter"}]
    
    
@schema
class PupilOracles(dj.Computed):
    definition = """
    -> ScanIdx
    -> PupilFilterMethod
    trial_id: int     # trial number
    ---
    pupil_clip: enum('madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h')
    pupil_trial: smallint
    pupil_trace: blob
    pupil_trace_norm: blob
    pupil_trace_diff: blob
    pupil_trace_diff_norm: blob
    pupil_oracle: float
    pupil_oracle_norm: float
    pupil_diff_oracle: float
    pupil_diff_oracle_norm: float
    """
    
    @property
    def key_source(self):
        return ScanIdx() * PupilFilterMethod()
    
    def _make_tuples(self, key):
        
        animal_id = (Animal() * Session() * ScanIdx() & key).fetch1('animal_id')
        session = (Animal() * Session() * ScanIdx() & key).fetch1('session')
        scan_idx = (Animal() * Session() * ScanIdx() & key).fetch1('scan_idx')
        scan_key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx}
        
        # Get oracle info

        stimulus_info = get_stimulus_info(scan_key)
        clip_hashes = [a[2]['condition_hash'] for a in stimulus_info if a[3]['type'] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        
        # Get pupil info
        
        pupil_trace = (pupil.FittedPupil.Circle & scan_key).fetch('radius')
        pupil_times = (pupil.Eye & scan_key).fetch('eye_time')[0]
        depth_num = np.unique((meso.ScanInfo.Field & scan_key).fetch('z')).shape[0]
        scan_times = (stimulus.BehaviorSync() & scan_key).fetch1('frame_times')[::depth_num]
        
        # Interpolate nan values in pupil signal
        
        nans, x = nan_helper(pupil_trace)
        pupil_trace[nans]= np.interp(x(nans), x(~nans), pupil_trace[~nans])
        
        # Remove DC from pupil trace
        pupil_trace = pupil_trace - np.mean(pupil_trace)
        
        if key['pupil_filter_method'] == 1:
            print(key['pupil_filter_method'])

            # DOWNSAMPLE PUPIL TRACE

            pupil_trace = interpolate(pupil_trace, pupil_times, scan_times)

            # DIFERENTIATE PUPIL TRACE

            pupil_trace_diff = np.diff(pupil_trace)

            # GET PUPIL TRACE SEGMENTS FOR ALL CLIPS

            clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']
            count = 1
            for k,stim_hash in zip(clip,oracle_hashes):

                stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]

                pupil_segments = []
                pupil_segments_norm = []
                pupil_segments_diff = []
                pupil_segments_diff_norm = []

                min_trace_len = 10000000

                trial_count = 1
                for n,(start,stop) in enumerate(stim_start_stop_times):

                    pupil_segments.append(pupil_trace[start:stop])
                    pupil_segments_norm.append(normalize_signal(pupil_trace[start:stop]))
                    pupil_segments_diff.append(pupil_trace_diff[start:stop])
                    pupil_segments_diff_norm.append(normalize_signal(pupil_trace_diff[start:stop]))

                    if min_trace_len > (stop-start):
                        min_trace_len = stop-start

                for n in range(len(pupil_segments)):
                    pupil_segments[n] = pupil_segments[n][0:min_trace_len]
                    pupil_segments_norm[n] = pupil_segments_norm[n][0:min_trace_len]
                    pupil_segments_diff[n] = pupil_segments_diff[n][0:min_trace_len]
                    pupil_segments_diff_norm[n] = pupil_segments_diff_norm[n][0:min_trace_len]

                pupil_oracle = calculate_oracle(pupil_segments)
                pupil_oracle_norm = calculate_oracle(pupil_segments_norm)
                pupil_diff_oracle = calculate_oracle(pupil_segments_diff)
                pupil_diff_oracle_norm = calculate_oracle(pupil_segments_diff_norm)

                for n in range(len(pupil_oracle)):
                    key['trial_id'] = count
                    key['pupil_trial'] = trial_count
                    key['pupil_clip'] = k

                    key['pupil_trace'] = pupil_segments[n]
                    key['pupil_trace_norm'] = pupil_segments_norm[n]
                    key['pupil_trace_diff'] = pupil_segments_diff[n]
                    key['pupil_trace_diff_norm'] = pupil_segments_diff_norm[n]

                    key['pupil_oracle'] = pupil_oracle[n]
                    key['pupil_oracle_norm'] = pupil_oracle_norm[n]
                    key['pupil_diff_oracle'] = pupil_diff_oracle[n]
                    key['pupil_diff_oracle_norm'] = pupil_diff_oracle_norm[n]

                    self.insert1(key)
                    count += 1
                    trial_count += 1
                    
                    
        if key['pupil_filter_method'] == 2:
            print(key['pupil_filter_method'])
        
            # FILTER PUPIL TRACE

            # Filter requirements.
            order = 5
            fs = len(pupil_times)/pupil_times[-1]       # sample rate, Hz
            cutoff = 1  # desired cutoff frequency of the filter, Hz

            # Get the filter coefficients.
            b, a = butter_lowpass(cutoff, fs, order)

            # Apply filter
            pupil_trace = scipy.signal.filtfilt(b, a, pupil_trace)

            # DOWNSAMPLE PUPIL TRACE

            pupil_trace = interpolate(pupil_trace, pupil_times, scan_times)

            # DIFERENTIATE PUPIL TRACE

            pupil_trace_diff = np.diff(pupil_trace)

            # GET PUPIL TRACE SEGMENTS FOR ALL CLIPS

            clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']
            count = 1
            for k,stim_hash in zip(clip,oracle_hashes):

                stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]

                pupil_segments = []
                pupil_segments_norm = []
                pupil_segments_diff = []
                pupil_segments_diff_norm = []

                min_trace_len = 10000000

                trial_count = 1
                for n,(start,stop) in enumerate(stim_start_stop_times):

                    pupil_segments.append(pupil_trace[start:stop])
                    pupil_segments_norm.append(normalize_signal(pupil_trace[start:stop]))
                    pupil_segments_diff.append(pupil_trace_diff[start:stop])
                    pupil_segments_diff_norm.append(normalize_signal(pupil_trace_diff[start:stop]))

                    if min_trace_len > (stop-start):
                        min_trace_len = stop-start

                for n in range(len(pupil_segments)):
                    pupil_segments[n] = pupil_segments[n][0:min_trace_len]
                    pupil_segments_norm[n] = pupil_segments_norm[n][0:min_trace_len]
                    pupil_segments_diff[n] = pupil_segments_diff[n][0:min_trace_len]
                    pupil_segments_diff_norm[n] = pupil_segments_diff_norm[n][0:min_trace_len]

                pupil_oracle = calculate_oracle(pupil_segments)
                pupil_oracle_norm = calculate_oracle(pupil_segments_norm)
                pupil_diff_oracle = calculate_oracle(pupil_segments_diff)
                pupil_diff_oracle_norm = calculate_oracle(pupil_segments_diff_norm)

                for n in range(len(pupil_oracle)):
                    key['trial_id'] = count
                    key['pupil_trial'] = trial_count
                    key['pupil_clip'] = k

                    key['pupil_trace'] = pupil_segments[n]
                    key['pupil_trace_norm'] = pupil_segments_norm[n]
                    key['pupil_trace_diff'] = pupil_segments_diff[n]
                    key['pupil_trace_diff_norm'] = pupil_segments_diff_norm[n]

                    key['pupil_oracle'] = pupil_oracle[n]
                    key['pupil_oracle_norm'] = pupil_oracle_norm[n]
                    key['pupil_diff_oracle'] = pupil_diff_oracle[n]
                    key['pupil_diff_oracle_norm'] = pupil_diff_oracle_norm[n]

                    self.insert1(key)
                    count += 1
                    trial_count += 1
                    
                    
        if key['pupil_filter_method'] == 3:
            print(key['pupil_filter_method'])
        
            # FILTER PUPIL TRACE

            # Filter requirements.
            fs = len(pupil_times)/pupil_times[-1]       # sample rate, Hz

            # Get filter tapers.
            taps = bandpass_firwin(9, 0.1, 1.0, fs, window='hamming')

            # Apply filter
            pupil_trace = scipy.signal.filtfilt(taps, 0.39, pupil_trace)

            # DOWNSAMPLE PUPIL TRACE

            pupil_trace = interpolate(pupil_trace, pupil_times, scan_times)

            # DIFERENTIATE PUPIL TRACE

            pupil_trace_diff = np.diff(pupil_trace)

            # GET PUPIL TRACE SEGMENTS FOR ALL CLIPS

            clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']
            count = 1
            for k,stim_hash in zip(clip,oracle_hashes):

                stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]

                pupil_segments = []
                pupil_segments_norm = []
                pupil_segments_diff = []
                pupil_segments_diff_norm = []

                min_trace_len = 10000000

                trial_count = 1
                for n,(start,stop) in enumerate(stim_start_stop_times):

                    pupil_segments.append(pupil_trace[start:stop])
                    pupil_segments_norm.append(normalize_signal(pupil_trace[start:stop]))
                    pupil_segments_diff.append(pupil_trace_diff[start:stop])
                    pupil_segments_diff_norm.append(normalize_signal(pupil_trace_diff[start:stop]))

                    if min_trace_len > (stop-start):
                        min_trace_len = stop-start

                for n in range(len(pupil_segments)):
                    pupil_segments[n] = pupil_segments[n][0:min_trace_len]
                    pupil_segments_norm[n] = pupil_segments_norm[n][0:min_trace_len]
                    pupil_segments_diff[n] = pupil_segments_diff[n][0:min_trace_len]
                    pupil_segments_diff_norm[n] = pupil_segments_diff_norm[n][0:min_trace_len]

                pupil_oracle = calculate_oracle(pupil_segments)
                pupil_oracle_norm = calculate_oracle(pupil_segments_norm)
                pupil_diff_oracle = calculate_oracle(pupil_segments_diff)
                pupil_diff_oracle_norm = calculate_oracle(pupil_segments_diff_norm)

                for n in range(len(pupil_oracle)):
                    key['trial_id'] = count
                    key['pupil_trial'] = trial_count
                    key['pupil_clip'] = k

                    key['pupil_trace'] = pupil_segments[n]
                    key['pupil_trace_norm'] = pupil_segments_norm[n]
                    key['pupil_trace_diff'] = pupil_segments_diff[n]
                    key['pupil_trace_diff_norm'] = pupil_segments_diff_norm[n]

                    key['pupil_oracle'] = pupil_oracle[n]
                    key['pupil_oracle_norm'] = pupil_oracle_norm[n]
                    key['pupil_diff_oracle'] = pupil_diff_oracle[n]
                    key['pupil_diff_oracle_norm'] = pupil_diff_oracle_norm[n]

                    self.insert1(key)
                    count += 1
                    trial_count += 1
        
        
    
@schema
class UnitOracles(dj.Computed):
    definition = """
    -> ScanIdx
    -> UnitTraceMethod
    oracle_id: int
    ---
    unit_id: int
    unit_clip: enum('madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h')
    unit_trial: smallint
    unit_trace: blob
    unit_trace_norm: blob
    unit_oracle              : float
    unit_oracle_norm         : float
    """
    
    @property
    def key_source(self):
        return ScanIdx() * UnitTraceMethod()
    
    def _make_tuples(self, key):
        print('Cutting out activity traces for', key)
        
        animal_id = (Animal() * Session() * ScanIdx() & key).fetch1('animal_id')
        session = (Animal() * Session() * ScanIdx() & key).fetch1('session')
        scan_idx = (Animal() * Session() * ScanIdx() & key).fetch1('scan_idx')
        scan_key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx}
        
        units = np.unique((meso.Activity.Trace & scan_key).fetch('unit_id'))
        
        # Get oracle info

        stimulus_info = get_stimulus_info(scan_key)
        clip_hashes = [a[2]['condition_hash'] for a in stimulus_info if a[3]['type'] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        
        
        if key['unit_trace_method'] == 1:
            print(key['unit_trace_method'])
            unit_count_m1 = 1

            for unit in units:
                
                unit_id = {'unit_id': unit}
                print(unit_id)

                fluorescence_trace = (meso.ScanSet.Unit() * meso.Fluorescence.Trace() & scan_key & unit_id).fetch1('trace')

                oracles_per_clip = []
                traces_per_clip = []
                clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']

                for k,stim_hash in zip(clip,oracle_hashes):
                    stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]
                    unit_traces = []
                    unit_traces_norm = []
                    min_trace_len = 10000000

                    count_m1 = 1
                    for n,(start,stop) in enumerate(stim_start_stop_times):
                        unit_traces.append(fluorescence_trace[start:stop])
                        unit_traces_norm.append(normalize_signal(fluorescence_trace[start:stop]))
                        if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                    for n in range(len(unit_traces)):
                        unit_traces[n] = unit_traces[n][0:min_trace_len]
                        unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]

                    oracle = calculate_oracle(unit_traces)
                    oracle_norm = calculate_oracle(unit_traces_norm)

                    for n in range(len(oracle)):
                        
                        key['oracle_id'] = unit_count_m1
                        key['unit_id'] = int(unit)
                        key['unit_clip'] = k
                        key['unit_trial'] = count_m1
                        key['unit_trace'] = unit_traces[n]
                        key['unit_trace_norm'] = unit_traces_norm[n]
                        key['unit_oracle'] = oracle[n]
                        key['unit_oracle_norm'] = oracle_norm[n]
                        self.insert1(key)
                        count_m1 += 1
                        unit_count_m1 += 1
                        
        if key['unit_trace_method'] == 2:
            print(key['unit_trace_method'])
            unit_count_m1 = 1

            for unit in units:
                
                unit_id = {'unit_id': unit}
                print(unit_id)

                signal = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')

                oracles_per_clip = []
                traces_per_clip = []
                clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']

                for k,stim_hash in zip(clip,oracle_hashes):
                    stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]
                    unit_traces = []
                    unit_traces_norm = []
                    min_trace_len = 10000000

                    count_m1 = 1
                    for n,(start,stop) in enumerate(stim_start_stop_times):
                        unit_traces.append(signal[start:stop])
                        unit_traces_norm.append(normalize_signal(signal[start:stop]))
                        if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                    for n in range(len(unit_traces)):
                        unit_traces[n] = unit_traces[n][0:min_trace_len]
                        unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]

                    oracle = calculate_oracle(unit_traces)
                    oracle_norm = calculate_oracle(unit_traces_norm)

                    for n in range(len(oracle)):
                        
                        key['oracle_id'] = unit_count_m1
                        key['unit_id'] = int(unit)
                        key['unit_clip'] = k
                        key['unit_trial'] = count_m1
                        key['unit_trace'] = unit_traces[n]
                        key['unit_trace_norm'] = unit_traces_norm[n]
                        key['unit_oracle'] = oracle[n]
                        key['unit_oracle_norm'] = oracle_norm[n]
                        self.insert1(key)
                        count_m1 += 1
                        unit_count_m1 += 1

                        
        if key['unit_trace_method'] == 3:
            print(key['unit_trace_method'])
            unit_count_m1 = 1

            for unit in units:
                
                unit_id = {'unit_id': unit}
                print(unit_id)

                signal = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')

                oracles_per_clip = []
                traces_per_clip = []
                clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']

                for k,stim_hash in zip(clip,oracle_hashes):
                    stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]
                    unit_traces = []
                    unit_traces_norm = []
                    min_trace_len = 10000000

                    count_m1 = 1
                    for n,(start,stop) in enumerate(stim_start_stop_times):
                        unit_traces.append(abs(savgol_filter(signal[start:stop],5,2)))
                        unit_traces_norm.append(normalize_signal(abs(savgol_filter(signal[start:stop],5,2))))
                        if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                    for n in range(len(unit_traces)):
                        unit_traces[n] = unit_traces[n][0:min_trace_len]
                        unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]

                    oracle = calculate_oracle(unit_traces)
                    oracle_norm = calculate_oracle(unit_traces_norm)

                    for n in range(len(oracle)):
                        
                        key['oracle_id'] = unit_count_m1
                        key['unit_id'] = int(unit)
                        key['unit_clip'] = k
                        key['unit_trial'] = count_m1
                        key['unit_trace'] = unit_traces[n]
                        key['unit_trace_norm'] = unit_traces_norm[n]
                        key['unit_oracle'] = oracle[n]
                        key['unit_oracle_norm'] = oracle_norm[n]
                        self.insert1(key)
                        count_m1 += 1
                        unit_count_m1 += 1

                        
@schema
class UnitOraclesSlidingWindow(dj.Computed):
    definition = """
    -> ScanIdx
    -> UnitTraceMethod
    unit_oracles_id: int
    ---
    unit_id: int
    unit_clip: enum('madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h')
    window: smallint
    sliding_oracles              : float
    sliding_oracles_norm         : float
    """
    
    @property
    def key_source(self):
        return ScanIdx() * UnitTraceMethod()
    
    def _make_tuples(self, key):
        print('Cutting out activity traces for', key)
        
        animal_id = (Animal() * Session() * ScanIdx() & key).fetch1('animal_id')
        session = (Animal() * Session() * ScanIdx() & key).fetch1('session')
        scan_idx = (Animal() * Session() * ScanIdx() & key).fetch1('scan_idx')
        scan_key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx}
        
        units = np.unique((meso.Activity.Trace & scan_key).fetch('unit_id'))
        
        # Get oracle info

        stimulus_info = get_stimulus_info(scan_key)
        clip_hashes = [a[2]['condition_hash'] for a in stimulus_info if a[3]['type'] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        
        
        if key['unit_trace_method'] == 1:
            print(key['unit_trace_method'])
            unit_count_m1 = 1
            
            sliding_window = 10
            step = 1
            
            for unit in units:
                
                unit_id = {'unit_id': unit}
                print(unit_id)

                fluorescence_trace = (meso.ScanSet.Unit() * meso.Fluorescence.Trace() & scan_key & unit_id).fetch1('trace')

                oracles_per_clip = []
                traces_per_clip = []
                clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']

                for k,stim_hash in zip(clip,oracle_hashes):
                    stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]
                    unit_traces = []
                    unit_traces_norm = []
                    min_trace_len = 10000000

                    for n,(start,stop) in enumerate(stim_start_stop_times):
                        unit_traces.append(fluorescence_trace[start:stop])
                        unit_traces_norm.append(normalize_signal(fluorescence_trace[start:stop]))
                        if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                    for n in range(len(unit_traces)):
                        unit_traces[n] = unit_traces[n][0:min_trace_len]
                        unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]
                    
                    start_frame = 0
                    n = len(unit_traces)
                    num_windows = n - sliding_window + 1

                    sliding_oracles = []
                    sliding_oracles_norm = []
                    count_m1 = 1
                    for j in range(num_windows): 
                        sliding_oracles.append(np.nanmean(calculate_oracle(unit_traces[start_frame : start_frame + sliding_window])))
                        sliding_oracles_norm.append(np.nanmean(calculate_oracle(unit_traces_norm[start_frame : start_frame + sliding_window])))
                        start_frame += step
                        
                    for n in range(len(sliding_oracles)):
                        
                        key['unit_oracles_id'] = unit_count_m1
                        key['unit_id'] = unit
                        key['unit_clip'] = k
                        key['window'] = count_m1
                        key['sliding_oracles'] = sliding_oracles[n]
                        key['sliding_oracles_norm'] = sliding_oracles_norm[n]
                        self.insert1(key)
                        count_m1 += 1
                        unit_count_m1 += 1
                        
                        
        if key['unit_trace_method'] == 2:
            print(key['unit_trace_method'])
            unit_count_m1 = 1
            
            sliding_window = 10
            step = 1
            
            for unit in units:
                
                unit_id = {'unit_id': unit}
                print(unit_id)

                signal = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')

                oracles_per_clip = []
                traces_per_clip = []
                clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']

                for k,stim_hash in zip(clip,oracle_hashes):
                    stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]
                    unit_traces = []
                    unit_traces_norm = []
                    min_trace_len = 10000000

                    for n,(start,stop) in enumerate(stim_start_stop_times):
                        unit_traces.append(signal[start:stop])
                        unit_traces_norm.append(normalize_signal(signal[start:stop]))
                        if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                    for n in range(len(unit_traces)):
                        unit_traces[n] = unit_traces[n][0:min_trace_len]
                        unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]
                    
                    start_frame = 0
                    n = len(unit_traces)
                    num_windows = n - sliding_window + 1

                    sliding_oracles = []
                    sliding_oracles_norm = []
                    count_m1 = 1
                    for j in range(num_windows): 
                        sliding_oracles.append(np.nanmean(calculate_oracle(unit_traces[start_frame : start_frame + sliding_window])))
                        sliding_oracles_norm.append(np.nanmean(calculate_oracle(unit_traces_norm[start_frame : start_frame + sliding_window])))
                        start_frame += step
                        
                    for n in range(len(sliding_oracles)):
                        
                        key['unit_oracles_id'] = unit_count_m1
                        key['unit_id'] = unit
                        key['unit_clip'] = k
                        key['window'] = count_m1
                        key['sliding_oracles'] = sliding_oracles[n]
                        key['sliding_oracles_norm'] = sliding_oracles_norm[n]
                        self.insert1(key)
                        count_m1 += 1
                        unit_count_m1 += 1
                        
                        
        if key['unit_trace_method'] == 3:
            print(key['unit_trace_method'])
            unit_count_m1 = 1
            
            sliding_window = 10
            step = 1
            
            for unit in units:
                
                unit_id = {'unit_id': unit}
                print(unit_id)

                signal = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')

                oracles_per_clip = []
                traces_per_clip = []
                clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']

                for k,stim_hash in zip(clip,oracle_hashes):
                    stim_start_stop_times = [(a[0],a[1]) for a in stimulus_info if a[2]['condition_hash'] == stim_hash]
                    unit_traces = []
                    unit_traces_norm = []
                    min_trace_len = 10000000

                    for n,(start,stop) in enumerate(stim_start_stop_times):
                        unit_traces.append(abs(savgol_filter(signal[start:stop],5,2)))
                        unit_traces_norm.append(normalize_signal(abs(savgol_filter(signal[start:stop],5,2))))
                        if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                    for n in range(len(unit_traces)):
                        unit_traces[n] = unit_traces[n][0:min_trace_len]
                        unit_traces_norm[n] = unit_traces_norm[n][0:min_trace_len]
                    
                    start_frame = 0
                    n = len(unit_traces)
                    num_windows = n - sliding_window + 1

                    sliding_oracles = []
                    sliding_oracles_norm = []
                    count_m1 = 1
                    for j in range(num_windows): 
                        sliding_oracles.append(np.nanmean(calculate_oracle(unit_traces[start_frame : start_frame + sliding_window])))
                        sliding_oracles_norm.append(np.nanmean(calculate_oracle(unit_traces_norm[start_frame : start_frame + sliding_window])))
                        start_frame += step
                        
                    for n in range(len(sliding_oracles)):
                        
                        key['unit_oracles_id'] = unit_count_m1
                        key['unit_id'] = unit
                        key['unit_clip'] = k
                        key['window'] = count_m1
                        key['sliding_oracles'] = sliding_oracles[n]
                        key['sliding_oracles_norm'] = sliding_oracles_norm[n]
                        self.insert1(key)
                        count_m1 += 1
                        unit_count_m1 += 1
                        
                        
             
def calculate_oracle(traces):
    traces = np.array(traces)
    X = []
    for i in range(len(traces)):
        X.append(i)
    correl = []
    for i in range(len(traces)):
        x = X[:]
        x.pop(i)
        mu = np.nanmean(traces[x], axis = 0)
        correl.append(corr(traces[i], mu))
    return correl
        
def corr(y1, y2, axis=-1, eps=1e-8, return_p=False, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final mean of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=1) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=1) + eps)
    if not return_p:
        return (y1 * y2).mean(axis=axis, **kwargs)
    else:
        rho = (y1 * y2).mean(axis=axis, **kwargs)
        N = y1.shape[axis] if not isinstance(axis, (tuple, list)) else np.prod([y1.shape[a] for a in axis])
        t = rho / np.sqrt((1 - rho ** 2) / (N - 2))
        prob = distributions.t.sf(np.abs(t), N - 1) * 2
        return rho, prob
           

def interpolate(to_be_interpolated, to_be_interpolated_times, interpolant_times):
   
    x = to_be_interpolated_times
    f = interp1d(x, to_be_interpolated, bounds_error=False, fill_value='extrapolate') # interpolate function
    xnew = interpolant_times
    trace = f(xnew)
        
    return trace    
    
def normalize_signal(signel, offset=0):
    signel = signel - np.nanmean(signel)
    signel = signel / (np.nanmax(signel) - np.nanmin(signel))
    signel = signel - np.nanmin(signel) + offset
    return signel    

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps