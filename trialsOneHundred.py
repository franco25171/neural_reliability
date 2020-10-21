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
    trial_idx: int     # trial number
    ---
    condition_hash: varchar(20) 
    stimulus_type: varchar(20)
    movie_name: varchar(20)
    clip_number: int
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
        
        # Get oracle info

        pupil_stimulus_info = get_pupil_stimulus_info(key)
        clip_hashes = [i[2]['hash'] for i in pupil_stimulus_info if i[2]['stimulus_type'] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        oracle_hashes_o, clip_names = find_pupil_oracle_hashes_order(pupil_stimulus_info, oracle_hashes) # Get oracle_hashes and movie_names in the order they were presented
        
        # Get pupil info
        
        pupil_trace = (pupil.FittedPupil.Circle & key).fetch('radius')
        pupil_times = (pupil.Eye & key).fetch('eye_time')[0]
        depth_num = np.unique((meso.ScanInfo.Field & key).fetch('z')).shape[0]
        scan_times = (stimulus.BehaviorSync() & key).fetch1('frame_times')[::depth_num]
        
        # Interpolate nan values in pupil signal
        
        nans, x = nan_helper(pupil_trace)
        pupil_trace[nans]= np.interp(x(nans), x(~nans), pupil_trace[~nans])
        
        # DOWNSAMPLE PUPIL TRACE
        pupil_trace = interpolate(pupil_trace, pupil_times, scan_times)
       
        # LOW PASS FILTER & HAMMING WINDOW BAND PASS FILTER REQUIREMENTS
        order = 5
        fs = len(pupil_times)/pupil_times[-1]       # sample rate, Hz
        cutoff = 1  # desired cutoff frequency of the filter, Hz
        b, a = butter_lowpass(cutoff, fs, order)    # Get the filter coefficients.
        taps = bandpass_firwin(9, 0.1, 1.0, fs, window='hamming')    # Get filter tapers.
        
        # GET PUPIL TRACE SEGMENTS FOR ALL CLIPS
        # clip = ['madmax', 'starwars', 'finalrun', 'bigrun', 'sports1m_ui', 'sports1m_7h']
        
        #min_trace_len = 10000000
        
        if key['pupil_filter_method'] == 1:
            print(key['pupil_filter_method'])
            
            # Differentiate pupil trace
            pupil_trace_diff = np.diff(pupil_trace)
             
            trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_pupil_segments, all_pupil_segments_norm, all_pupil_segments_diff, all_pupil_segments_diff_norm, all_pupil_oracle, all_pupil_oracle_norm, all_pupil_diff_oracle, all_pupil_diff_oracle_norm = cut_out_pupil_segments(pupil_stimulus_info, clip_names, oracle_hashes_o, pupil_trace, pupil_trace_diff)

            for n in range(len(all_pupil_segments)):
                key['trial_idx'] = int(trial_idx[n])
                key['condition_hash'] = condition_hash[n]
                key['stimulus_type'] = stimulus_type[n]
                key['movie_name'] = movie_name[n]
                key['clip_number'] = int(clip_number[n])

                key['pupil_trace'] = all_pupil_segments[n]
                key['pupil_trace_norm'] = all_pupil_segments_norm[n]
                key['pupil_trace_diff'] = all_pupil_segments_diff[n]
                key['pupil_trace_diff_norm'] = all_pupil_segments_diff_norm[n]

                key['pupil_oracle'] = all_pupil_oracle[n]
                key['pupil_oracle_norm'] = all_pupil_oracle_norm[n]
                key['pupil_diff_oracle'] = all_pupil_diff_oracle[n]
                key['pupil_diff_oracle_norm'] = all_pupil_diff_oracle_norm[n]

                self.insert1(key)
                    
                    
        if key['pupil_filter_method'] == 2:
            print(key['pupil_filter_method'])
       
            # Apply filter
            pupil_trace = scipy.signal.filtfilt(b, a, pupil_trace)

            # Differentiate pupil trace

            pupil_trace_diff = np.diff(pupil_trace)

            trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_pupil_segments, all_pupil_segments_norm, all_pupil_segments_diff, all_pupil_segments_diff_norm, all_pupil_oracle, all_pupil_oracle_norm, all_pupil_diff_oracle, all_pupil_diff_oracle_norm = cut_out_pupil_segments(pupil_stimulus_info, clip_names, oracle_hashes_o, pupil_trace, pupil_trace_diff)

            for n in range(len(all_pupil_segments)):
                key['trial_idx'] = int(trial_idx[n])
                key['condition_hash'] = condition_hash[n]
                key['stimulus_type'] = stimulus_type[n]
                key['movie_name'] = movie_name[n]
                key['clip_number'] = int(clip_number[n])

                key['pupil_trace'] = all_pupil_segments[n]
                key['pupil_trace_norm'] = all_pupil_segments_norm[n]
                key['pupil_trace_diff'] = all_pupil_segments_diff[n]
                key['pupil_trace_diff_norm'] = all_pupil_segments_diff_norm[n]

                key['pupil_oracle'] = all_pupil_oracle[n]
                key['pupil_oracle_norm'] = all_pupil_oracle_norm[n]
                key['pupil_diff_oracle'] = all_pupil_diff_oracle[n]
                key['pupil_diff_oracle_norm'] = all_pupil_diff_oracle_norm[n]

                self.insert1(key)
                   
                    
        if key['pupil_filter_method'] == 3:
            print(key['pupil_filter_method'])
        
            # Apply hamming window band-pass filter
            pupil_trace = scipy.signal.filtfilt(taps, 0.39, pupil_trace)

            # Differentiate pupil trace

            pupil_trace_diff = np.diff(pupil_trace)

            trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_pupil_segments, all_pupil_segments_norm, all_pupil_segments_diff, all_pupil_segments_diff_norm, all_pupil_oracle, all_pupil_oracle_norm, all_pupil_diff_oracle, all_pupil_diff_oracle_norm = cut_out_pupil_segments(pupil_stimulus_info, clip_names, oracle_hashes_o, pupil_trace, pupil_trace_diff)

            for n in range(len(all_pupil_segments)):
                key['trial_idx'] = int(trial_idx[n])
                key['condition_hash'] = condition_hash[n]
                key['stimulus_type'] = stimulus_type[n]
                key['movie_name'] = movie_name[n]
                key['clip_number'] = int(clip_number[n])

                key['pupil_trace'] = all_pupil_segments[n]
                key['pupil_trace_norm'] = all_pupil_segments_norm[n]
                key['pupil_trace_diff'] = all_pupil_segments_diff[n]
                key['pupil_trace_diff_norm'] = all_pupil_segments_diff_norm[n]

                key['pupil_oracle'] = all_pupil_oracle[n]
                key['pupil_oracle_norm'] = all_pupil_oracle_norm[n]
                key['pupil_diff_oracle'] = all_pupil_diff_oracle[n]
                key['pupil_diff_oracle_norm'] = all_pupil_diff_oracle_norm[n]

                self.insert1(key)
        
        
    
@schema
class UnitTraces(dj.Computed):
    definition = """
    -> ScanIdx
    -> UnitTraceMethod
    unit_id: int
    trial_idx: int
    --- 
    condition_hash: varchar(20)
    stimulus_type: varchar(20)
    movie_name: varchar(20)
    clip_number: int
    unit_trace: blob
    unit_trace_norm: blob
    """
    
    @property
    def key_source(self):
        return ScanIdx() * UnitTraceMethod()
    
    def _make_tuples(self, key):
        print('Cutting out activity traces for', key)
        
        scan_key = (ScanIdx() & key).fetch1('KEY')
        
        units_stimulus_info = get_units_stimulus_info(scan_key)
        
        # Get oracle info

        clip_hashes = [i[3]['hash'] for i in units_stimulus_info[0] if i[3]['stimulus_type'] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        oracle_hashes_o, clip_names = find_oracle_hashes_order(units_stimulus_info, oracle_hashes) # Get oracle_hashes and movie_names in the order they were presented
        units = [i[0][2]['unit_id'] for i in units_stimulus_info] # Get all units ids
        
        min_trace_len = 10000000
        
        if key['unit_trace_method'] == 1:
            print(key['unit_trace_method'])

            for i in range(len(units)):
                
                unit_id = {'unit_id': units[i]}
                print(unit_id)
                
                unit_trace = (meso.ScanSet.Unit() * meso.Fluorescence.Trace() & scan_key & unit_id).fetch1('trace')

                trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_unit_traces, all_unit_traces_norm = cut_out_unit_segments(units_stimulus_info, clip_names, oracle_hashes_o, unit_trace, i, units)

                for n in range(len(all_unit_traces)):

                    key['unit_id'] = (units[i])
                    key['trial_idx'] = int(trial_idx[n])
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = int(clip_number[n])
                    key['unit_trace'] = all_unit_traces[n]
                    key['unit_trace_norm'] = all_unit_traces_norm[n]
                    self.insert1(key)
                        
        if key['unit_trace_method'] == 2:
            print(key['unit_trace_method'])
            unit_count_m1 = 1

            for i in range(len(units)):
                
                unit_id = {'unit_id': units[i]}
                print(unit_id)

                unit_trace = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')

                trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_unit_traces, all_unit_traces_norm = cut_out_unit_segments(units_stimulus_info, clip_names, oracle_hashes_o, unit_trace, i, units)

                for n in range(len(all_unit_traces)):

                    key['unit_id'] = int(units[i])
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['unit_trace'] = all_unit_traces[n]
                    key['unit_trace_norm'] = all_unit_traces_norm[n]
                    self.insert1(key)

                        
        if key['unit_trace_method'] == 3:
            print(key['unit_trace_method'])
            unit_count_m1 = 1

            for i in range(len(units)):
                
                unit_id = {'unit_id': units[i]}
                print(unit_id)

                unit_trace = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')
                unit_trace = abs(savgol_filter(unit_trace, 5, 2))

                trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_unit_traces, all_unit_traces_norm = cut_out_unit_segments(units_stimulus_info, clip_names, oracle_hashes_o, unit_trace, i, units)
                             
                for n in range(len(all_unit_traces)):

                    key['unit_id'] = int(units[i])
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['unit_trace'] = all_unit_traces[n]
                    key['unit_trace_norm'] = all_unit_traces_norm[n]
                    self.insert1(key)
                    
                    
@schema
class UnitOracles(dj.Computed):
    definition = """
    -> UnitTraces
    unit_id: int
    trial_idx: int
    --- 
    condition_hash: varchar(20)
    stimulus_type: varchar(20)
    movie_name: varchar(20)
    clip_number: int
    unit_oracle              : float
    unit_oracle_norm         : float
    """
    
    def _make_tuples(self, key):
        print('Cutting out activity traces for', key)
        
        scan_key = (ScanIdx() & key).fetch1('KEY')
        
        units_stimulus_info = get_units_stimulus_info(scan_key)
        
        # Get oracle info

        clip_hashes = [i[3]['hash'] for i in units_stimulus_info[0] if i[3]['stimulus_type'] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        oracle_hashes_o, clip_names = find_oracle_hashes_order(units_stimulus_info, oracle_hashes) # Get oracle_hashes and movie_names in the order they were presented
        units = [i[0][2]['unit_id'] for i in units_stimulus_info] # Get all units ids
        
        if key['unit_trace_method'] == 1:
            print(key['unit_trace_method'])
            unit_trace_method = key['unit_trace_method']

            for i in range(len(units)):
                
                unit_id = {'unit_id': units[i]}
                print(unit_id)
                
                #unit_trace = (meso.ScanSet.Unit() * meso.Fluorescence.Trace() & scan_key & unit_id).fetch1('trace')

                trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_oracles, all_oracles_norm = calculate_unit_oracles(units_stimulus_info, clip_names, oracle_hashes_o, i, units, unit_id, unit_trace_method)

                for n in range(len(all_oracles)):

                    key['unit_id'] = (units[i])
                    key['trial_idx'] = int(trial_idx[n])
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = int(clip_number[n])
                    key['unit_oracle'] = all_oracles[n]
                    key['unit_oracle_norm'] = all_oracles_norm[n]
                    self.insert1(key)
                        
        if key['unit_trace_method'] == 2:
            print(key['unit_trace_method'])
            unit_trace_method = key['unit_trace_method']

            for i in range(len(units)):
                
                unit_id = {'unit_id': units[i]}
                print(unit_id)

                #unit_trace = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')

                trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_oracles, all_oracles_norm = calculate_unit_oracles(units_stimulus_info, clip_names, oracle_hashes_o, i, units, unit_id, unit_trace_method)

                for n in range(len(all_oracles)):

                    key['unit_id'] = int(units[i])
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['unit_oracle'] = all_oracles[n]
                    key['unit_oracle_norm'] = all_oracles_norm[n]
                    self.insert1(key)

                        
        if key['unit_trace_method'] == 3:
            print(key['unit_trace_method'])
            unit_trace_method = key['unit_trace_method']

            for i in range(len(units)):
                
                unit_id = {'unit_id': units[i]}
                print(unit_id)

                #unit_trace = (meso.Activity.Trace & scan_key & unit_id).fetch1('trace')
                #unit_trace = abs(savgol_filter(unit_trace, 5, 2))

                trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_oracles, all_oracles_norm = calculate_unit_oracles(units_stimulus_info, clip_names, oracle_hashes_o, i, units, unit_id, unit_trace_method)
                             
                for n in range(len(all_oracles)):

                    key['unit_id'] = int(units[i])
                    key['trial_idx'] = trial_idx[n]
                    key['condition_hash'] = condition_hash[n]
                    key['stimulus_type'] = stimulus_type[n]
                    key['movie_name'] = movie_name[n]
                    key['clip_number'] = clip_number[n]
                    key['unit_oracle'] = all_oracles[n]
                    key['unit_oracle_norm'] = all_oracles_norm[n]
                    self.insert1(key)
                        
                        
             
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

def get_pupil_stimulus_info(key):
    
    if np.unique((stimulus.Trial & key).fetch('trial_idx')).shape[0] != len(stimulus.Trial & key):
        raise PipelineException("Error: Duplicate trial indices detected. Is the key unique for one scan?") 
    
    # Get stimulus information for all trials

    all_trials, all_hashes, all_flips, all_conditions, all_movie_names, all_clip_numbers = ((stimulus.Trial * 
                                                                                             stimulus.Condition * 
                                                                                             stimulus.Clip & key).fetch('trial_idx',
                                                                                                                             'condition_hash',
                                                                                                                             'flip_times',
                                                                                                                             'stimulus_type',
                                                                                                                             'movie_name',
                                                                                                                             'clip_number',
                                                                                                                             order_by='last_flip ASC'))
    trial_information = []
    for trial, hashes, trial_condition, movie_name, clip_number in zip(all_trials, all_hashes, all_conditions, all_movie_names, all_clip_numbers):

        if trial_condition == 'stimulus.Monet2':
            fps, duration, version = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet2).fetch1('fps', 'duration', 'stimulus_version')
            extra_info = {'type': trial_condition, 'fps': fps, 'duration': duration, 'version': version}
        elif trial_condition == 'stimulus.Trippy':
            fps, duration, version = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('fps', 'duration', 'stimulus_version')
            extra_info = {'type': trial_condition, 'fps': fps, 'duration': duration, 'version': version}
        elif trial_condition == 'stimulus.Clip':
            extra_info = {'trial_id': trial, 'hash': hashes, 'stimulus_type': trial_condition, 'movie_name': movie_name, 'clip_number': clip_number}

            trial_information.append(extra_info)

    # determine field offset for ScanImage frame times
    all_z = (meso.ScanInfo.Field & key).fetch('z', order_by='field ASC')

    # Number of depths recorded from during scan
    slice_num = len(all_z)

    frame_times = (stimulus.Sync & key).fetch1('frame_times')
    field_offset = 0
    frame_times = frame_times[field_offset::slice_num] # Sliced ScanImage times for a single depth of interest
    start_idx = []
    end_idx = []
    pupil_stimulus_info = []
    for j in range(len(all_flips)):
        trial_flips = all_flips[j]
        start_idx.append((frame_times < trial_flips.squeeze()[0]).sum()) 
        end_idx.append((frame_times < trial_flips.squeeze()[-1]).sum())
        pupil_stimulus_info.append([start_idx[j], end_idx[j], trial_information[j]])
        
    return pupil_stimulus_info

def get_units_stimulus_info(key):
    
    if np.unique((stimulus.Trial & key).fetch('trial_idx')).shape[0] != len(stimulus.Trial & key):
        raise PipelineException("Error: Duplicate trial indices detected. Is the key unique for one scan?") 
    
    # Get stimulus information for all trials

    all_trials, all_hashes, all_flips, all_conditions, all_movie_names, all_clip_numbers = ((stimulus.Trial * 
                                                                                             stimulus.Condition * 
                                                                                             stimulus.Clip & key).fetch('trial_idx',
                                                                                                                             'condition_hash',
                                                                                                                             'flip_times',
                                                                                                                             'stimulus_type',
                                                                                                                             'movie_name',
                                                                                                                             'clip_number',
                                                                                                                             order_by='last_flip ASC'))
    trial_information = []
    for trial, hashes, trial_condition, movie_name, clip_number in zip(all_trials, all_hashes, all_conditions, all_movie_names, all_clip_numbers):

        if trial_condition == 'stimulus.Monet2':
            fps, duration, version = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Monet2).fetch1('fps', 'duration', 'stimulus_version')
            extra_info = {'type': trial_condition, 'fps': fps, 'duration': duration, 'version': version}
        elif trial_condition == 'stimulus.Trippy':
            fps, duration, version = ((stimulus.Trial & trial_key) * stimulus.Condition * stimulus.Trippy).fetch1('fps', 'duration', 'stimulus_version')
            extra_info = {'type': trial_condition, 'fps': fps, 'duration': duration, 'version': version}
        elif trial_condition == 'stimulus.Clip':
            extra_info = {'trial_id': trial, 'hash': hashes, 'stimulus_type': trial_condition, 'movie_name': movie_name, 'clip_number': clip_number}

            trial_information.append(extra_info)

    # determine field offset for ScanImage frame times
    all_z = (meso.ScanInfo.Field & key).fetch('z', order_by='field ASC')

    # Number of depths recorded from during scan
    slice_num = len(all_z)

    # Get keys from all units
    unit_keys, fields, ms_delays, field_z = (meso.ScanSet.UnitInfo * meso.Activity.Trace() * meso.ScanInfo.Field & key).fetch('KEY', 'field', 'ms_delay', 'z', order_by='unit_id ASC')
    frame_times = (stimulus.Sync & key).fetch1('frame_times')
    units_stimulus_info = []
    for i in range(5):#len(unit_keys)):
        print(i)   
        field_offset = np.where(all_z == field_z[i])[0][0]
        frame_times_wdelay = frame_times + ms_delays[i]/1000 # ScanImage times on stimulus clock
        frame_times_wdelay = frame_times_wdelay[field_offset::slice_num] # Sliced ScanImage times for a single depth of interest
        start_idx = []
        end_idx = []
        trial_info = []
        for j in range(len(all_flips)):
            trial_flips = all_flips[j]
            start_idx.append((frame_times_wdelay < trial_flips.squeeze()[0]).sum()) 
            end_idx.append((frame_times_wdelay < trial_flips.squeeze()[-1]).sum())
            trial_info.append([start_idx[j], end_idx[j], {'unit_id': unit_keys[i]['unit_id']}, trial_information[j]])
        units_stimulus_info.append(trial_info)
        
    return units_stimulus_info


def find_pupil_oracle_hashes_order(pupil_stimulus_info, oracle_hashes):

    clip_names = []
    for i in pupil_stimulus_info:
        clip_names.append([[i[0],i[1],i[2]['hash'],i[2]['movie_name'],i[2]['clip_number']] 
                           for clip_hash in oracle_hashes if i[2]['hash'] == clip_hash])

    clip_names = [i for i in clip_names if i]
    clip_names = np.squeeze(clip_names[0:len(oracle_hashes)])

    oracle_hashes_o = [i[2] for i in clip_names]
    movie_names = [i[3]+'_'+i[4] for i in clip_names]
    
    return oracle_hashes_o, movie_names


def find_oracle_hashes_order(units_stimulus_info, oracle_hashes):

    clip_names = []
    for i in units_stimulus_info[0]:
        clip_names.append([[i[0],i[1],i[3]['hash'],i[3]['movie_name'],i[3]['clip_number']] 
                           for clip_hash in oracle_hashes if i[3]['hash'] == clip_hash])

    clip_names = [i for i in clip_names if i]
    clip_names = np.squeeze(clip_names[0:len(oracle_hashes)])

    oracle_hashes_o = [i[2] for i in clip_names]
    movie_names = [i[3]+'_'+i[4] for i in clip_names]
    
    return oracle_hashes_o, movie_names


def cut_out_pupil_segments(pupil_stimulus_info, clip_names, oracle_hashes_o, pupil_trace, pupil_trace_diff):
    
    all_pupil_segments = [0] * len(pupil_stimulus_info)
    all_pupil_segments_norm = [0] * len(pupil_stimulus_info)
    all_pupil_segments_diff = [0] * len(pupil_stimulus_info)
    all_pupil_segments_diff_norm = [0] * len(pupil_stimulus_info)
    all_pupil_oracle = [0] * len(pupil_stimulus_info)
    all_pupil_oracle_norm = [0] * len(pupil_stimulus_info)
    all_pupil_diff_oracle = [0] * len(pupil_stimulus_info)
    all_pupil_diff_oracle_norm = [0] * len(pupil_stimulus_info)
    trial_idx = [0] * len(pupil_stimulus_info)
    condition_hash = [0] * len(pupil_stimulus_info)
    stimulus_type = [0] * len(pupil_stimulus_info)
    movie_name = [0] * len(pupil_stimulus_info)
    clip_number = [0] * len(pupil_stimulus_info)
    for clip,stim_hash in zip(clip_names,oracle_hashes_o):
        stim_start_stop_times = [(a[0],a[1],a[2]['trial_id'],a[2]['hash'],a[2]['stimulus_type'],
                                  a[2]['movie_name'],a[2]['clip_number']) for a in 
                                 pupil_stimulus_info if a[2]['hash'] == stim_hash]

        pupil_segments = []
        pupil_segments_norm = []
        pupil_segments_diff = []
        pupil_segments_diff_norm = []
        min_trace_len = 10000000

        for n,(start,stop,index,condition,stype,mname,cnumber) in enumerate(stim_start_stop_times):

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
            
            all_pupil_segments[stim_start_stop_times[n][2]] = pupil_segments[n]
            all_pupil_segments_norm[stim_start_stop_times[n][2]] = pupil_segments_norm[n]
            all_pupil_segments_diff[stim_start_stop_times[n][2]] = pupil_segments_diff[n]
            all_pupil_segments_diff_norm[stim_start_stop_times[n][2]] = pupil_segments_diff_norm[n]
            trial_idx[stim_start_stop_times[n][2]] = stim_start_stop_times[n][2] 
            condition_hash[stim_start_stop_times[n][2]] = stim_start_stop_times[n][3]
            stimulus_type[stim_start_stop_times[n][2]] = stim_start_stop_times[n][4]
            movie_name[stim_start_stop_times[n][2]] = stim_start_stop_times[n][5]
            clip_number[stim_start_stop_times[n][2]] = stim_start_stop_times[n][6]

        pupil_oracle = calculate_oracle(pupil_segments)
        pupil_oracle_norm = calculate_oracle(pupil_segments_norm)
        pupil_diff_oracle = calculate_oracle(pupil_segments_diff)
        pupil_diff_oracle_norm = calculate_oracle(pupil_segments_diff_norm)
        
        for n in range(len(pupil_segments)):
            all_pupil_oracle[stim_start_stop_times[n][2]] = pupil_oracle[n]
            all_pupil_oracle_norm[stim_start_stop_times[n][2]] = pupil_oracle_norm[n]
            all_pupil_diff_oracle[stim_start_stop_times[n][2]] = pupil_diff_oracle[n]
            all_pupil_diff_oracle_norm[stim_start_stop_times[n][2]] = pupil_diff_oracle_norm[n]
            
    return trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_pupil_segments, all_pupil_segments_norm, all_pupil_segments_diff, all_pupil_segments_diff_norm, all_pupil_oracle, all_pupil_oracle_norm, all_pupil_diff_oracle, all_pupil_diff_oracle_norm  
            
            
def cut_out_unit_segments(units_stimulus_info, clip_names, oracle_hashes_o, unit_trace, i, units):
                                        
    all_unit_traces = [0] * len(units_stimulus_info[0])
    all_unit_traces_norm = [0] * len(units_stimulus_info[0])
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
            
    return trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_unit_traces, all_unit_traces_norm

def calculate_unit_oracles(units_stimulus_info, clip_names, oracle_hashes_o, i, units, unit_id, unit_trace_method):
                                        
    all_oracles = [0] * len(units_stimulus_info[0])
    all_oracles_norm = [0] * len(units_stimulus_info[0])
    trial_idx = [0] * len(units_stimulus_info[0])
    condition_hash = [0] * len(units_stimulus_info[0])
    stimulus_type = [0] * len(units_stimulus_info[0])
    movie_name = [0] * len(units_stimulus_info[0])
    clip_number = [0] * len(units_stimulus_info[0])

    for clip,stim_hash in zip(clip_names,oracle_hashes_o):
        stim_start_stop_times = [(a[3]['trial_id'],a[3]['hash'],a[3]['stimulus_type'],
                                  a[3]['movie_name'],a[3]['clip_number']) for a in 
                                 units_stimulus_info[i] if a[2]['unit_id'] == units[i] and a[3]['hash'] == stim_hash]

        condition_hash_ = {'condition_hash': stim_hash}
        unit_trace_method_ = {'unit_trace_method': unit_trace_method}
        unit_traces = (UnitTraces() & unit_trace_method_ & unit_id & condition_hash_).fetch('unit_trace')
        unit_traces_norm = (UnitTraces() & unit_trace_method_ & unit_id & condition_hash_).fetch('unit_trace_norm')

        for n in range(len(unit_traces)):
            trial_idx[stim_start_stop_times[n][0]] = stim_start_stop_times[n][0] 
            condition_hash[stim_start_stop_times[n][0]] = stim_start_stop_times[n][1]
            stimulus_type[stim_start_stop_times[n][0]] = stim_start_stop_times[n][2]
            movie_name[stim_start_stop_times[n][0]] = stim_start_stop_times[n][3]
            clip_number[stim_start_stop_times[n][0]] = stim_start_stop_times[n][4]
            
        unit_traces_ = []
        unit_traces_norm_ = []
        for n in range(len(unit_traces)):
            unit_traces_.append(unit_traces[n])
            unit_traces_norm_.append(unit_traces_norm[n])

        oracle = calculate_oracle(unit_traces_)
        oracle_norm = calculate_oracle(unit_traces_norm_)

        for n in range(len(unit_traces)):
            all_oracles[stim_start_stop_times[n][0]] = oracle[n]
            all_oracles_norm[stim_start_stop_times[n][0]] = oracle_norm[n]
            
    return trial_idx, condition_hash, stimulus_type, movie_name, clip_number, all_oracles, all_oracles_norm
