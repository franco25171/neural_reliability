import datajoint as dj
import numpy as np
from pipeline import pupil, meso, treadmill, experiment
#from stimulus import stimulus
from stimulus.utils import get_stimulus_info
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from sklearn.linear_model import LinearRegression
import scipy as spy
from scipy.signal import find_peaks
import math
import scipy
from scipy.signal import savgol_filter
from scipy.signal import firwin
stimulus = dj.create_virtual_module('stim', 'pipeline_stimulus')

schema = dj.schema('franco_100_trials')

@schema
class PupilFilterMethod(dj.Lookup):
    definition = """
    # Variants in pupil traces
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
class UnitOracleMethod(dj.Lookup):
    definition = """
    # Variants in how unit oracle is calculated 
    unit_oracle_method               : tinyint              # method index
    ---
    description              : varchar(250)         # description of method
    """

    contents = [{'unit_oracle_method':1, 'description':"oracle is calculated for every trial correlating a particular trial with the mean of the rest of the trials (leave one out)"}]
    
@schema
class PupilOracleMethod(dj.Lookup):
    definition = """
    # Variants in how pupil oracle is calculated
    pupil_oracle_method               : tinyint              # method index
    ---
    description              : varchar(250)         # description of method
    """

    contents = [{'pupil_oracle_method':1, 'description':"oracle is calculated for every trial correlating a particular trial with the mean of the rest of the trials (leave one out)"}]

@schema 
class OracleScans(dj.Manual):
    definition = """
    -> experiment.Scan
    ---
    """

@schema 
class TrialInfo(dj.Computed):
    definition = """
    -> OracleScans
    -> stimulus.Trial
    ---
    condition_hash: varchar(20)
    stimulus_type: varchar(20)
    movie_name: varchar(20)
    clip_number: int
    flip_times: blob
    """
         
    def _make_tuples(self, key):
        print(key)
        
        flip_times = (TrialInfo() & key).fetch('flip_times', order_by='trial_idx')
        
        frame_times = (stimulus.Sync & key).fetch1('frame_times')
        
        condition_hash, flip_times, stimulus_type, movie_name, clip_number = ((stimulus.Trial * stimulus.Condition * stimulus.Clip & key).fetch('condition_hash', 'flip_times','stimulus_type','movie_name','clip_number'))
        
        # Calculate movie clip's start and end times for every trial
        #time = [(frame_times < flip_times[0].squeeze()[0]).sum(), (frame_times < flip_times[0].squeeze()[-1]).sum()]

        key['condition_hash'] = condition_hash[0]
        key['stimulus_type'] = stimulus_type[0]
        key['movie_name'] = movie_name[0]
        key['clip_number'] = int(clip_number[0])
        key['flip_times'] = flip_times[0]
        self.insert1(key)
       
    
@schema            
class OracleClip(dj.Computed):
    definition = """
    -> OracleScans
    oracle_hash: varchar(20)
    --- 
    number_of_repeats: int     # Number of times a particular hash was repeated
    """ 
    
    def _make_tuples(self, key):

        trial_idx, condition_hash = (TrialInfo().fetch('trial_idx', 'condition_hash'))
        unique_hashes = np.unique(condition_hash, return_counts=True)
        
        for unique_hash, count in zip(unique_hashes[0],unique_hashes[1]):
            if count > 9:
                
                key['oracle_hash'] = unique_hash
                key['number_of_repeats'] = count
                self.insert1(key)
                
                
@schema            
class UnitTrace(dj.Computed):
    definition = """
    -> OracleScans
    -> meso.ScanInfo
    -> UnitTraceMethod
    --- 
    
    """  
    
    class Trial(dj.Part):
        definition = """ # Cut out unit traces corresponding to oracle clips
        -> UnitTrial
        -> meso.ScanSet.Unit
        -> TrialInfo    
        ---
        condition_hash: varchar(20)
        stimulus_type: varchar(20)
        movie_name: varchar(20)
        clip_number: int
        trace: blob
        trace_norm: blob
        """
        
    def make(self, key):
        print(key)
        
        scan_key = (OracleScans() & key).fetch1('KEY')
        
        oracle_hashes = (OracleClip().fetch('oracle_hash'))

        units, ms_delays, fields_z, segmentation_methods = ((OracleScans() * meso.Activity.Trace() * meso.ScanSet.UnitInfo() * meso.ScanInfo.Field() & key).fetch('unit_id', 'ms_delay', 'z', 'segmentation_method', order_by='unit_id ASC'))
        self.insert1(key)
        
        all_z = (meso.ScanInfo.Field & key).fetch('z', order_by='field ASC')
        slice_num = len(all_z) # Number of depths recorded from during scan
        trial_idx, flip_times, condition_hash, stimulus_type, movie_name, clip_number = (TrialInfo() & key).fetch('trial_idx', 'flip_times', 'condition_hash', 'stimulus_type', 'movie_name', 'clip_number', order_by='trial_idx')
        frame_times = (stimulus.Sync & key).fetch1('frame_times')
        
        for unit, ms_delay, field_z, segmentation_method in zip(units, ms_delays, fields_z, segmentation_methods):
            print(unit)
        
            field_offset = np.where(all_z == field_z)[0][0]
            frame_times_wdelay = frame_times + ms_delay/1000 # ScanImage times on stimulus clock
            frame_times_wdelay = frame_times_wdelay[field_offset::slice_num] # Sliced ScanImage times for a single depth of interest

            unit_traces = []
            unit_traces_norm = []
            min_trace_len = 10000000

            start_idx = []
            end_idx = []
            for flip_time in flip_times:
                start_idx.append((frame_times_wdelay < flip_time.squeeze()[0]).sum())
                end_idx.append((frame_times_wdelay < flip_time.squeeze()[-1]).sum())


            if key['unit_trace_method'] == 1:
                
                unit_id = {'unit_id': unit}
                unit_trace = (meso.ScanSet.Unit() * meso.Fluorescence.Trace() & scan_key & unit_id).fetch1('trace')
                
                for start, stop in zip(start_idx, end_idx):
                    unit_traces.append(unit_trace[start:stop])
                    unit_traces_norm.append(normalize_signal(unit_trace[start:stop]))
                    if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                
                for n in range(len(unit_traces)):
                    if condition_hash[n] in oracle_hashes:
                        key['unit_id'] = unit
                        key['segmentation_method'] = segmentation_method
                        key['trial_idx'] = trial_idx[n]
                        key['condition_hash'] = condition_hash[n]
                        key['stimulus_type'] = stimulus_type[n]
                        key['movie_name'] = movie_name[n]
                        key['clip_number'] = clip_number[n]
                        key['trace'] = unit_traces[n][0:min_trace_len]
                        key['trace_norm'] = unit_traces_norm[n][0:min_trace_len]
                        UnitTrial.Trace.insert1(key)
                        
            if key['unit_trace_method'] == 2:
                
                unit_id = {'unit_id': unit}
                unit_trace = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')
                
                for start, stop in zip(start_idx, end_idx):
                    unit_traces.append(unit_trace[start:stop])
                    unit_traces_norm.append(normalize_signal(unit_trace[start:stop]))
                    if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                
                for n in range(len(unit_traces)):
                    if condition_hash[n] in oracle_hashes:
                        key['unit_id'] = unit
                        key['segmentation_method'] = segmentation_method
                        key['trial_idx'] = trial_idx[n]
                        key['condition_hash'] = condition_hash[n]
                        key['stimulus_type'] = stimulus_type[n]
                        key['movie_name'] = movie_name[n]
                        key['clip_number'] = clip_number[n]
                        key['trace'] = unit_traces[n][0:min_trace_len]
                        key['trace_norm'] = unit_traces_norm[n][0:min_trace_len]
                        UnitTrial.Trace.insert1(key)
                        
            if key['unit_trace_method'] == 3:
                
                unit_id = {'unit_id': unit}
                unit_trace = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')
                unit_trace = abs(savgol_filter(unit_trace, 5, 2))
                
                for start, stop in zip(start_idx, end_idx):
                    unit_traces.append(unit_trace[start:stop])
                    unit_traces_norm.append(normalize_signal(unit_trace[start:stop]))
                    if min_trace_len > (stop-start):
                            min_trace_len = stop-start
                
                for n in range(len(unit_traces)):
                    if condition_hash[n] in oracle_hashes:
                        key['unit_id'] = unit
                        key['segmentation_method'] = segmentation_method
                        key['trial_idx'] = trial_idx[n]
                        key['condition_hash'] = condition_hash[n]
                        key['stimulus_type'] = stimulus_type[n]
                        key['movie_name'] = movie_name[n]
                        key['clip_number'] = clip_number[n]
                        key['trace'] = unit_traces[n][0:min_trace_len]
                        key['trace_norm'] = unit_traces_norm[n][0:min_trace_len]
                        UnitTrial.Trace.insert1(key)
                    
         
                   
@schema            
class PupilRadius(dj.Computed):
    definition = """
    -> OracleScans
    -> pupil.Eye
    -> PupilFilterMethod
    
    --- 
    """
    
    class Trial(dj.Part):
        definition = """ # Cut out pupil traces corresponding to oracle clips
        -> PupilRadius
        -> TrialInfo
        ---
        condition_hash: varchar(20)
        stimulus_type: varchar(20)
        movie_name: varchar(20)
        clip_number: int
        radius: blob
        radius_diff: blob
        """
        
    def make(self, key):
        print(key)
        
        oracle_hashes = ((OracleClip() & key).fetch('oracle_hash'))

        pupil_trace = (pupil.FittedPupil.Circle() & key).fetch('radius')
        pupil_times = (pupil.Eye() & key).fetch('eye_time')[0]
        pupil_times_ = pupil_times.copy()
        depth_num = np.unique((meso.ScanInfo.Field & key).fetch('z')).shape[0]
        scan_times = (stimulus.BehaviorSync() & key).fetch1('frame_times')[::depth_num]
        self.insert1(key)
        
        # Interpolate nan values in pupil signal
        nans, x = nan_helper(pupil_trace)
        pupil_trace[nans]= np.interp(x(nans), x(~nans), pupil_trace[~nans])
        pupil_times_[nans] = np.interp(x(nans), x(~nans), pupil_times_[~nans])
        
        # Downsample pupil trace
        pupil_trace = interpolate(pupil_trace, pupil_times, scan_times)
        
        # Low pass filter & hamming window band pass filter requirements  
        order = 5
        fs = len(pupil_times)/pupil_times_[-1]       # sample rate, Hz
        print(fs)
        cutoff = 1  # desired cutoff frequency of the filter, Hz
        b, a = butter_lowpass(cutoff, fs, order)    # Get the filter coefficients.
        taps = bandpass_firwin(9, 0.1, 1.0, fs, window='hamming')    # Get filter tapers.
        
        # Get stimulus info
        all_z = np.unique((meso.ScanInfo.Field & key).fetch('z', order_by='field ASC'))
        slice_num = len(all_z) # Number of depths recorded from during scan
        trial_idx, flip_times, condition_hash, stimulus_type, movie_name, clip_number = (TrialInfo() & key).fetch('trial_idx', 'flip_times', 'condition_hash', 'stimulus_type', 'movie_name', 'clip_number', order_by='trial_idx')
        frame_times = (stimulus.Sync & key).fetch1('frame_times')
        
        field_offset = 0 #np.where(all_z == field_z)[0][0]
        frame_times = frame_times[field_offset::slice_num] # Sliced ScanImage times for a single depth of interest
        
        pupil_traces = []
        pupil_traces_diff = []
        min_trace_len = 10000000
        
        start_idx = []
        end_idx = []
        for flip_time in flip_times:
            start_idx.append((frame_times < flip_time.squeeze()[0]).sum())
            end_idx.append((frame_times < flip_time.squeeze()[-1]).sum())
            
        if key['pupil_filter_method'] == 1:
            
            # Differentiate pupil trace
            pupil_trace_diff = np.diff(pupil_trace)
            
            for start, stop in zip(start_idx, end_idx):
                pupil_traces.append(pupil_trace[start:stop])
                pupil_traces_diff.append(pupil_trace_diff[start:stop])
                if min_trace_len > (stop-start):
                        min_trace_len = stop-start

            for n in range(len(pupil_traces)):
                if condition_hash[n] in oracle_hashes:
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['radius'] = pupil_traces[n][0:min_trace_len]
                    key['radius_diff'] = pupil_traces_diff[n][0:min_trace_len]
                    PupilRadius.Trial.insert1(key)
                    
        if key['pupil_filter_method'] == 2:
            
            # Apply low pass filter
            pupil_trace = scipy.signal.filtfilt(b, a, pupil_trace)
            
            # Differentiate pupil trace
            pupil_trace_diff = np.diff(pupil_trace)
            
            for start, stop in zip(start_idx, end_idx):
                pupil_traces.append(pupil_trace[start:stop])
                pupil_traces_diff.append(pupil_trace_diff[start:stop])
                if min_trace_len > (stop-start):
                        min_trace_len = stop-start

            for n in range(len(pupil_traces)):
                if condition_hash[n] in oracle_hashes:
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['radius'] = pupil_traces[n][0:min_trace_len]
                    key['radius_diff'] = pupil_traces_diff[n][0:min_trace_len]
                    PupilRadius.Trial.insert1(key)
                    
        if key['pupil_filter_method'] == 3:
            
            # Apply hamming window band-pass filter
            pupil_trace = scipy.signal.filtfilt(taps, 0.39, pupil_trace)
            
            # Differentiate pupil trace
            pupil_trace_diff = np.diff(pupil_trace)
            
            for start, stop in zip(start_idx, end_idx):
                pupil_traces.append(pupil_trace[start:stop])
                pupil_traces_diff.append(pupil_trace_diff[start:stop])
                if min_trace_len > (stop-start):
                        min_trace_len = stop-start

            for n in range(len(pupil_traces)):
                if condition_hash[n] in oracle_hashes:
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['radius'] = pupil_traces[n][0:min_trace_len]
                    key['radius_diff'] = pupil_traces_diff[n][0:min_trace_len]
                    PupilRadius.Trial.insert1(key)

                    

        
@schema            
class UnitOracle(dj.Computed):
    definition = """ # Contains the average oracle for every unit
    -> OracleScans
    -> UnitTraceMethod
    -> UnitOracleMethod
    -> meso.ScanSet.Unit
    --- 
    unit_oracle: float      # average oracle for all trials in a unit
    """
    
    class Trial(dj.Part):
        definition = """ # Contains the oracle for every unit trial
        -> UnitOracle
        -> TrialInfo
        ---
        unit_trial_oracle: float   # oracle for every trial in a unit
        """
                    
    def make(self, key):
        print(key)
        
        tuple_ = key.copy()
        
        unit, unit_traces, trial_idx = (UnitTrial.Trace() * UnitOracleMethod() * meso.ScanSet.Unit() & key).fetch('unit_id', 'trace', 'trial_idx', order_by='trial_idx ASC')
        unit = np.unique(unit)

        unit_traces_ = []
        for unit_trace in unit_traces:
            unit_traces_.append(unit_trace)

        oracles = calculate_oracle(unit_traces_)
        unit_mean_oracle = np.mean(oracles)

        key['unit_oracle'] = unit_mean_oracle
        self.insert1(key)

        for oracle, trial in zip(oracles, trial_idx):
            tuple_['trial_idx'] = trial 
            tuple_['unit_trial_oracle'] = oracle
            UnitOracle.Trial.insert1(tuple_)
        
              
        
@schema            
class PupilOracle(dj.Computed):
    definition = """ # Contains the oracle for every pupil trial
    -> PupilRadius
    -> PupilOracleMethod
    --- 
    pupil_oracle: float      # average pupil oracle for all trials in a scan
    """
    
    class Trial(dj.Part):
        definition = """ # Contains the oracle for every pupil trial
        -> PupilOracle
        -> TrialInfo
        ---
        pupil_trial_oracle: float   # oracle for every trial
        """
                
    def make(self, key):
        print(key)
        
        tuple_ = key.copy()

        pupil_traces, trial_idx = (PupilRadius.Trial() * PupilOracleMethod() & key).fetch('radius', 'trial_idx', order_by='trial_idx ASC')
        
        pupil_traces_ = []
        for pupil_trace in pupil_traces:
            pupil_traces_.append(pupil_trace)

        oracles = calculate_oracle(pupil_traces_)
        pupil_mean_oracle = np.mean(oracles)

        key['pupil_oracle'] = pupil_mean_oracle
        self.insert1(key)
        
        for oracle, trial in zip(oracles, trial_idx):
            tuple_['trial_idx'] = trial 
            tuple_['pupil_trial_oracle'] = oracle
            PupilOracle.Trial.insert1(tuple_)
            
    
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

def interpolate(to_be_interpolated, to_be_interpolated_times, interpolant_times):
   
    x = to_be_interpolated_times
    f = interp1d(x, to_be_interpolated, bounds_error=False, fill_value='extrapolate') # interpolate function
    xnew = interpolant_times
    trace = f(xnew)
        
    return trace 

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
    
