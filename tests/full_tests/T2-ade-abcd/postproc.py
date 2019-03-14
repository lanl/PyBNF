

def postprocess(simdata):
    """
    Make an arbitrary edit to the simulation data, so we can check the postprocessing call works. 
    """
    mycol = simdata.cols['B']
    simdata.data[4,mycol] = 123456.
    
    return simdata
