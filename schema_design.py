import datajoint as dj
import numpy as np
from pipeline import pupil, meso, treadmill
from stimulus import stimulus
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
class TrialInfo(dj.Computed):
    definition = """
    -> ScanIdx
    trial_idx: int
    ---
    condition_hash: varchar(20)
    flip_times: blob
    stimulus_type: varchar(20)
    movie_name: varchar(20)
    clip_number: int
    """
    
    def _make_tuples(self, key):
        
        trial_idx, condition_hash, flip_times, stimulus_type, movie_name, clip_number = ((stimulus.Trial * stimulus.Condition * stimulus.Clip & key).fetch('trial_idx', 'condition_hash', 'flip_times','stimulus_type','movie_name','clip_number', order_by='last_flip ASC'))
        
        for i in trial_idx:
        
            key['trial_idx'] = trial_idx[i]
            key['condition_hash'] = condition_hash[i]
            key['flip_times'] = flip_times[i]
            key['stimulus_type'] = stimulus_type[i]
            key['movie_name'] = movie_name[i]
            key['clip_number'] = int(clip_number[i])
            self.insert1(key)
       
    
@schema            
class OracleClip(dj.Computed):
    definition = """
    -> ScanIdx
    trial_idx: int
    --- 
    oracle_hash: varchar(20)
    """ 
    
    def _make_tuples(self, key):
        
        trial_idx, condition_hash, flip_times, stimulus_type, movie_name, clip_number = (TrialInfo() & key).fetch('trial_idx', 'condition_hash', 'flip_times', 'stimulus_type', 'movie_name', 'clip_number')
        clip_hashes = [i[1] for i in zip(trial_idx, condition_hash, stimulus_type) if i[2] == 'stimulus.Clip']
        unique_hashes = np.unique(clip_hashes, return_counts=True)
        oracle_hashes_ = unique_hashes[0][np.where(unique_hashes[1] > 99)]
        
        data = [(i[0], i[1], i[2], i[3], i[4], i[5]) for i in zip(trial_idx, condition_hash, flip_times, stimulus_type, movie_name, clip_number)]
        
        for i in data:
            if i[1] in oracle_hashes_:
                
                key['trial_idx'] = i[0]
                key['oracle_hash'] = i[1]
                self.insert1(key)
                
                
@schema            
class UnitTrial(dj.Computed):
    definition = """
    -> ScanIdx
    unit_id: int
    --- 
    
    """    
    
    class Trace(dj.Part):
        definition = """ # Cut out unit traces corresponding to oracle clips
        -> UnitTrial
        -> TrialInfo
        -> UnitTraceMethod
        ---
        trace: blob
        """
        
@schema            
class PupilTrial(dj.Computed):
    definition = """
    -> ScanIdx
    trial_idx: int
    --- 
    """
    
    class Trace(dj.Part):
        definition = """ # Cut out pupil traces corresponding to oracle clips
        -> PupilTrial
        -> TrialInfo
        -> PupilFilterMethod
        ---
        trace: blob
        """
        
@schema            
class UnitOracle(dj.Computed):
    definition = """ # Contains the average oracle for every unit
    -> UnitTrial
    -> UnitTraceMethod
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
        
        
@schema            
class PupilOracle(dj.Computed):
    definition = """ # Contains the oracle for every pupil trial
    -> ScanIdx
    -> PupilFilterMethod
    --- 
    pupil_oracle: float      # average pupil oracle for all trials in a scan
    """
    
    class Trial(dj.Part):
        definition = """ # Contains the oracle for every pupil trial
        -> PupilTrial
        -> PupilOracle
        -> TrialInfo
        ---
        pupil_trial_oracle: float   # oracle for every trial
        """