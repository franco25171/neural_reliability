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
class OracleScans(dj.Computed):
    definition = """
    -> experiment.Scan
    ---
    """
    
@schema 
class TrialInfo(dj.Computed):
    definition = """
    -> stimulus.Trial
    ---
    """
    
@schema            
class OracleClip(dj.Computed):
    definition = """
    -> OracleScans
    oracle_hash: varchar(20)
    --- 
    number_of_repeats: int     # Number of times a particular hash was repeated
    """ 
    
@schema            
class UnitTrial(dj.Computed):
    definition = """
    -> experiment.Scan
    -> UnitTraceMethod
    --- 
    
    """  
    
    class Trace(dj.Part):
        definition = """ # Cut out unit traces corresponding to oracle clips
        -> UnitTrial
        -> TrialInfo
        unit_id: int
        ---
        trace: blob
        """
        
    @schema            
class PupilTrial(dj.Computed):
    definition = """
    -> experiment.Scan
    -> PupilFilterMethod
    --- 
    """
    
    class Radius(dj.Part):
        definition = """ # Cut out pupil traces corresponding to oracle clips
        -> PupilTrial
        -> TrialInfo
        ---
        trace: blob
        """
        
@schema            
class UnitOracle(dj.Computed):
    definition = """ # Contains the average oracle for every unit
    -> UnitTrial
    -> UnitOracleMethod
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
    -> PupilTrial
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
        
    
