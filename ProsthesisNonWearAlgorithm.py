# Non-wear algorithm
import numpy as np
import pandas as pd
import math

def zero_runs(a):
    # create an array that is 1 where a is 0, and pad each end with an extra 0
    iszero = np.concatenate(([0],np.equal(a,0).astype(np.int8),[0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1,2)
    return ranges

def ProsthesisNonWearAlgorithm(VectorMagnitude_Prosthesis, EPOCH):
    amplitudeSensitivity = 15 # how tall can a spike in a non-wear period be?
    gapSensitivity = 5 # how many minutes do there need to be between spikes in non-wear period?
    # This non-wear algorithm is presented in Appendix 7b of Alix Chadwell's PhD thesis. 
    # As the categorisation of a count as wear/non-wear depends upon the categorisation of the previous count, this requires a loop to be run.
    # This has been edited to do some pre-allocation first for speed so may differ slightly from the original algorithm

    ## Initialise wear status

    Wear = (1,)
    NonWear = (0,)
    WearStatus = np.empty((len(VectorMagnitude_Prosthesis)))
    WearStatus[:] = np.nan
    
    #### Pre-allocation step 1:
    # If points are supposedly activity (greater than the *'amplitude threshold'*) and there is less than *'gap sensitivity'* before the next spike of supposed activity, then assume those points to be wear.
    # - Find all of the indexes where the vector magnitude was greater than the amplitude threshold
    # - Check how long between each point and the next one
    # - If there was less than gap sensitivity time between the points then allocate the first point as wear
  
    PossibleWear_Idx = VectorMagnitude_Prosthesis.loc[VectorMagnitude_Prosthesis>amplitudeSensitivity].index
    PossibleWear_Idx_lower = tuple(PossibleWear_Idx) + (len(VectorMagnitude_Prosthesis)+1,)
    PossibleWear_Idx_upper = (0,) + tuple(PossibleWear_Idx)
    differenceBetweenPossibleWearIdx = pd.Series(tuple(map(lambda i, j: i - j, PossibleWear_Idx_lower, PossibleWear_Idx_upper)))
    #PossibleWear_Idx_lower = pd.Series(PossibleWear_Idx_lower) ## Is this row needed?
    Idx_ToKeep = (differenceBetweenPossibleWearIdx <= gapSensitivity*(60/EPOCH)).drop(0) ## .drop(0) get's rid of the extra index at the start I think (need to double check and remind myself).
    Wear_Idx = PossibleWear_Idx[Idx_ToKeep]
    WearStatus[Wear_Idx] = Wear[0]

   
    # print('After preallocation 1 (obvious wear):')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))
    
    #### Pre-allocation step 2:
    # If points are supposedly not activity (equal to zero) and are followed by 20 mins (inclusive) of points below the *'amplitude threshold'*, then assume those points to be non-wear
    # - Exclude any rows where the wear status has already been filled in to save time (by setting the value equal to amplitude sensitivity +1)
    # - Find all of the indexes where the vector magnitude is less than or equal to the amplitude threshold
    # - Set these points equal to zero to make the data easier to work with
    # - Detect the runs of zeros (data below the amplitude threshold)
    # - Separate out the runs where the first point has a vector magnitude of zero
    # - Count how many zeros in each run
    # - If more than or equal to 20 mins of zeros in a row, the points before the last 20 mins (-1 epoch) are equal to non-wear
    VM_thresholded = VectorMagnitude_Prosthesis[:].copy()
    AlreadyFilled = ~np.isnan(WearStatus)*(amplitudeSensitivity+1) # use this to exclude the already filled rows from the analysis
    VM_thresholded = VM_thresholded + AlreadyFilled
    VM_thresholded[VM_thresholded<= amplitudeSensitivity] = 0 ## I've shifted this down 2 rows, I think that is still correct
    
    runs = zero_runs(VM_thresholded)
    DoesRunStartAboveZero = VectorMagnitude_Prosthesis[runs[:, 0]]==0
    runs = runs[DoesRunStartAboveZero]

    noOfZeros = runs[:,1]-runs[:,0]

    # Only pre-allocate if 20 mins of no activity - easier
    ZeroIdx_ToKeep = (noOfZeros >= 20*60/EPOCH)

    Start_Idx = runs[ZeroIdx_ToKeep][:, 0]
    Stop_Idx = runs[ZeroIdx_ToKeep][:, 1] - 20*60/EPOCH
    #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
    #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION

    def multi_arange(Start_Idx, Stop_Idx): 
        # np.arange returns evenly spaced values within a given interval...
        # multi_arange is designed to do this for multiple rows
        steps = np.ones(len(Start_Idx)).astype(int)
        lens = ((Stop_Idx-Start_Idx) + steps-np.sign(steps))//steps
        Start_Idx = Start_Idx[lens>0]
        Stop_Idx = Stop_Idx[lens>0]
        steps = np.ones(len(Start_Idx)).astype(int)
        lens = ((Stop_Idx-Start_Idx) + steps-np.sign(steps))//steps
        b = np.repeat(steps, lens)
        ends = (lens-1)*steps + Start_Idx
        b[0] = Start_Idx[0]
        b[lens[:-1].cumsum()] = Start_Idx[1:] - ends[:-1]
        return b.cumsum()
      
    NonWear_Idx = multi_arange(Start_Idx, Stop_Idx.astype(int))
    WearStatus[NonWear_Idx] = NonWear[0]
    
    # print('After preallocation 2 (obvious non-wear):')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))

    #### Pre-allocation step 3:
    # Allocate the first point in the recording if it hasn't already been set in the first 2 steps assuming non-wear pre-recording.
    # - If there is activity on the first point then assume wear, else if there is no activity assume non-wear
    # print('Wear status of point 1:', WearStatus[0])
    if math.isnan(WearStatus[0]): # Initialise to assume non-wear pre recording if the first count is 0.
        if VectorMagnitude_Prosthesis[0] == 0:
            WearStatus[0] = NonWear[0]
        else:
            WearStatus[0] = Wear[0]
        
    # print('After preallocation 3 (1st data point):')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))

    #### Pre-allocation step 4:
    # If points have a vector magnitude below the *'amplitude threshold'* AND follow a known non-wear period then assume those points to be non-wear.
    # - Exclude any rows where the wear status has already been filled in to save time (by setting the value equal to amplitude sensitivity +1)
    # - Find all of the indexes where the vector magnitude is less than or equal to the amplitude threshold
    # - Set these points equal to zero to make the data easier to work with
    # - Detect the runs of zeros (data below the amplitude threshold)
    # - Compare each run to wear status of the previous point, those which are part of a non-wear run can be assumed to be potential non-wear. **Could I do something later to deal with the iterative ones where you maybe have W W W W ? 0 ? 0 0 0 0 0 0 0 0 where the 1st ? would end up classified as wear, but the next ? could become non-wear. But is ignored as the run started on the 1st ? ... maybe this needs to be considered as a new run for each value above zero but below the threshold? But then bits would be double counted.**
    VM_thresholded = VectorMagnitude_Prosthesis[:].copy()
    AlreadyFilled = ~np.isnan(WearStatus)*(amplitudeSensitivity+1) # use this to exclude the already filled rows from the analysis
    VM_thresholded = VM_thresholded + AlreadyFilled
    VM_thresholded[VM_thresholded<= amplitudeSensitivity] = 0 ## I've shifted this down 2 rows, I think that is still correct
    runs = zero_runs(VM_thresholded)
    PreviousWearStatus = WearStatus[runs[:,0]-1]
    runs = runs[PreviousWearStatus==0] # runs following non-wear

    Start_Idx = runs[:, 0]
    Stop_Idx = runs[:, 1]
    #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
    #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION

    NonWear_Idx = multi_arange(Start_Idx, Stop_Idx.astype(int))
    WearStatus[NonWear_Idx] = NonWear[0]
    
    #print(WearStatus[0:20])
    #print(VectorMagnitude_Prosthesis[0:20])
    # print('After preallocation 4 (less obvious non-wear):')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))
    
    #### Pre-allocation step 5:
    # If points fall during a known wear period and there is less than *'gap sensitivity'* before the next spike of supposed activity, then assume those points to be wear.
    # - Identify the runs which have not been allocated yet and set them equal to zero
    # - Identify which of these runs follow a wear period
    
    VM_thresholded = VectorMagnitude_Prosthesis[:].copy()
    SetNotFilledZero = ~np.isnan(WearStatus)
    VM_thresholded = (VM_thresholded + 1) * SetNotFilledZero # +1 to make sure we don't catch VM=0 already allocated as non-wear
    runsOfUnAllocated = zero_runs(VM_thresholded)
    PreviousWearStatus = WearStatus[runsOfUnAllocated[:,0]-1]
    runsFollowingWear = runsOfUnAllocated[PreviousWearStatus==1]
        
    FindSpikes = np.ones(len(VectorMagnitude_Prosthesis))
    Start_Idx = runsFollowingWear[:, 0]
    Stop_Idx = runsFollowingWear[:, 1]
    #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
    #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION
    PossibleWearRuns_Idx = multi_arange(Start_Idx, Stop_Idx.astype(int))
    FindSpikes[PossibleWearRuns_Idx] = 0
    Spikes_Idx = VectorMagnitude_Prosthesis.loc[VectorMagnitude_Prosthesis>amplitudeSensitivity].index
    FindSpikes[Spikes_Idx] = 1
    
    runsBetweenSpikes = zero_runs(FindSpikes)

    SpikeRunLengths = runsBetweenSpikes[:,1]-runsBetweenSpikes[:,0]

    SpikeRunsToKeep = runsBetweenSpikes[SpikeRunLengths<= gapSensitivity*(60/EPOCH)]

    Start_Idx = SpikeRunsToKeep[:, 0]
    Stop_Idx = SpikeRunsToKeep[:, 1] + 1
    #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
    #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION   
        
    PossibleWear_Idx = multi_arange(Start_Idx, Stop_Idx.astype(int))
        
    # Check if all the idx being kept are part of a run which started at the start Idx - e.g. not the second half of a run with a ? in the middle
    Run_Idx_lower = tuple(PossibleWear_Idx) + (len(VectorMagnitude_Prosthesis)+1,)
    Run_Idx_upper = (0,) + tuple(PossibleWear_Idx)
    differenceBetweenRunIdx = pd.Series(tuple(map(lambda i, j: i - j, Run_Idx_lower, Run_Idx_upper)))
    if((PossibleWear_Idx[(differenceBetweenRunIdx>1).iloc[:-1]][0]) != (PossibleWear_Idx[0])):
        RunStartIdx = np.insert(PossibleWear_Idx[(differenceBetweenRunIdx>1).iloc[:-1]],0, PossibleWear_Idx[0])
    else:
        RunStartIdx = PossibleWear_Idx[(differenceBetweenRunIdx>1).iloc[:-1]]

    RunEndIdx = PossibleWear_Idx[(differenceBetweenRunIdx>1).drop(0)]
    if PossibleWear_Idx[-1] == len(VM_thresholded):
        RunEndIdx = np.append(RunEndIdx,len(VM_thresholded))
    elif PossibleWear_Idx[-1] >= len(VM_thresholded)-gapSensitivity:
        RunEndIdx[-1] = len(VM_thresholded)
    FullRunStartIdx = (np.in1d(RunStartIdx, runsFollowingWear[:,0])) # Mini run starts that match big run starts
    FullRunStart = RunStartIdx[FullRunStartIdx] # Mini run starts
    FullRunEnd = RunEndIdx[FullRunStartIdx] # Mini run starts equivalent ends
    RunsToKeep = np.array([FullRunStart, FullRunEnd], dtype=object).T
    
    ####################
    ## Note that I think this misses one value every time at the end.....try and fix if needed to speed up.
    ####################

    Start_Idx = RunsToKeep[:, 0]
    Stop_Idx = RunsToKeep[:, 1]
    #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
    #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION

    NonWear_Idx = multi_arange(Start_Idx.astype(int), Stop_Idx.astype(int))
    WearStatus[NonWear_Idx] = Wear[0]
    
    # print('After preallocation 5 (even less obvious wear):')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))

    #### Pre-allocation step 6:
    # If points have a vector magnitude above zero AND follow a known wear period then assume those points to be wear.
    # - Exclude any rows where the wear status has already been filled in to save time (by setting the value equal to zero)
    # - Find all of the indexes where the vector magnitude is greater than zero
    # - Set these points equal to zero to make the data easier to work with (making sure to swap all the ones which were already zero to ones)
    # - Detect the runs of zeros (data below the amplitude threshold)
    # - Compare each run to wear status of the previous point, those which are part of a wear run can be assumed to be potential wear. **Could I do something later to deal with the iterative ones - see step 4??**
    VM_thresholded = VectorMagnitude_Prosthesis[:].copy()
    SetAlreadyFilledZero = np.isnan(WearStatus) # use this to exclude the already filled rows from the analysis
    VM_thresholded = VM_thresholded * SetAlreadyFilledZero
    VM_thresholded2 = VM_thresholded[:].copy()
    VM_thresholded2[VM_thresholded == 0] = 1
    VM_thresholded2[VM_thresholded > 0] = 0
    runs = zero_runs(VM_thresholded2)
    PreviousWearStatus = WearStatus[runs[:,0]-1]
    runs = runs[PreviousWearStatus==1] # runs following wear

    Start_Idx = runs[:, 0]
    Stop_Idx = runs[:, 1]
    #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
    #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION

    NonWear_Idx = multi_arange(Start_Idx, Stop_Idx.astype(int))
    WearStatus[NonWear_Idx] = Wear[0]
    
    #print(WearStatus[0:20])
    #print(VectorMagnitude_Prosthesis[0:20])
    # print('After preallocation 6 (less obvious wear):')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))
    
    #### Iteration once all pre-allocation possible has been done
    for num, val in enumerate(VectorMagnitude_Prosthesis):
        if math.isnan(WearStatus[num]):
            #print(num, val, WearStatus[num], WearStatus[num-1])
            if val == 0:                               # if current count is 0
                if WearStatus[num-1] == 0:   # was previous count categorised as non wear?
                    WearStatus[num] = NonWear[0]  # if yes, current count is also categorised as non wear
            else:
                if WearStatus[num-1] == 1:   # was previous count categorised as wear?
                    WearStatus[num] = Wear[0]    # if yes, current count is also categorised as wear
                elif val <= amplitudeSensitivity:      # if no, is the current count less than the amplitude threshold? 
                    WearStatus[num] = NonWear[0]  # (it's unlikely that they just put the monitor/prosthesis on)
            if math.isnan(WearStatus[num]):
                # if wear status hasn't yet been determined, we are looking at a zero count following a wear period, 
                # or a count above the sensitivity threshold following a non-wear period. 
                # Look at the trends for the next 20 minutes to decide.
                Next20mins = VectorMagnitude_Prosthesis[num+1:num+20*int(60/EPOCH)+1]
                if len(Next20mins) < 20: # If less than 20 mins left we will make the assumption that the wear status has not changed.
                    WearStatus[num] = WearStatus[num-1]
                elif max(Next20mins) <= amplitudeSensitivity: # are the next 20 mins all below the amplitude threshold?
                    WearStatus[num] = NonWear[0]       # if yes, current count is categorised as non wear
                else:
                    IndexWithActivity = tuple(Next20mins[Next20mins > amplitudeSensitivity].index) + (len(Next20mins)+1,)
                    PriorIndexWithActivity = (0,) + tuple(Next20mins[Next20mins > amplitudeSensitivity].index)
                    differenceBetweenIndexes = tuple(map(lambda i, j: i - j, IndexWithActivity, PriorIndexWithActivity))
                    if min(differenceBetweenIndexes) > gapSensitivity*(60/EPOCH):
                        WearStatus[num] = NonWear[0]
                    else:
                        WearStatus[num] = Wear[0]

    # print('After iteration:')
    # print('Not Allocated:', np.count_nonzero(np.isnan(WearStatus)))
    # print('Allocated:', np.count_nonzero(~np.isnan(WearStatus)))   
    
    #### Secondary iteration
    # Find out how long each period of wear/non-wear is. If it is shorter than 10 mins reclassify it providing the subsequent classification block is not shorter.
    # - Use zero runs to find out the length of the runs (the ones will need to be swapped to zeros for this).
    # - Find the runs that are shorter than 10 mins
    # - Check for each run whether the next run is shorter
    # - If not swap the classification of the run
    def replaceWearStatusSingleRow(RunRow, WearStatus, ChangeCounter):
        Start_Idx = RunRow[0]
        Stop_Idx = RunRow[1]
        #### GETTING AN ERROR WHEN THESE ARE EQUAL TO ZERO!!
        #### DO AN IF NOT EMPTY CHECK BEFORE RUNNING FUNCTION
        Idx = np.arange(Start_Idx, Stop_Idx)
        WearStatus[Idx] = np.absolute(RunRow[2] - 1)
        ChangeCounter += 1
        return WearStatus, ChangeCounter

    ChangeCounter = 0

    NonWear_Runs = zero_runs(WearStatus)
    Wear_Runs = zero_runs(WearStatus-1)
    NonWear_RunLengths = (NonWear_Runs[:, 1] - NonWear_Runs[:, 0]).reshape((-1,1))
    Wear_RunLengths = (Wear_Runs[:, 1] - Wear_Runs[:, 0]).reshape((-1,1))
    ShortNonWear_Idx = np.where(NonWear_RunLengths<10*60/EPOCH)[0] # Find blocks shorter than 10 mins
    ShortWear_Idx = np.where(Wear_RunLengths<10*60/EPOCH)[0] # Find blocks shorter than 10 mins
    ShortNonWear_Run = NonWear_Runs[ShortNonWear_Idx]
    ShortWear_Run = Wear_Runs[ShortWear_Idx]
    LabelledNonWear_Runs = np.insert(ShortNonWear_Run, 2, 0, axis=1)
    LabelledWear_Runs = np.insert(ShortWear_Run, 2, 1, axis=1)

    # print(LabelledNonWear_Runs)
    # print(LabelledWear_Runs)

    ShortRuns = np.concatenate((LabelledNonWear_Runs, LabelledWear_Runs), axis=0)
    ShortRuns = ShortRuns[ShortRuns[:, 0].argsort()]
    #print(ShortRuns)

    for rows in range(len(ShortRuns)):
        if rows == len(ShortRuns)-1: # last instance
            [WearStatus, ChangeCounter] = replaceWearStatusSingleRow(ShortRuns[rows,:], WearStatus, ChangeCounter)
        elif ShortRuns[rows,1]==ShortRuns[rows+1,0]: # next section is also a short run
            if (ShortRuns[rows, 1] - ShortRuns[rows, 0])<=(ShortRuns[rows+1, 1] - ShortRuns[rows+1, 0]): # current run is shorter
                [WearStatus, ChangeCounter] = replaceWearStatusSingleRow(ShortRuns[rows,:], WearStatus, ChangeCounter)
            # else if next run is shorter do nothing - Note that this could mess up if you get more than 2 consecutive short periods in a row that each get shorter
        else:
            [WearStatus, ChangeCounter] = replaceWearStatusSingleRow(ShortRuns[rows,:], WearStatus, ChangeCounter)

    # print('After secondary iteration (re-allocation):')
    # print('Reallocated:', ChangeCounter, 'Runs')

    return(WearStatus) 