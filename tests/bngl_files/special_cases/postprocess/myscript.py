
def postprocess(data):
    """
    Multiply column x by 2.
    """
    
    col = data.cols['x']
    data.data[:, col] = 2 * data.data[:, col]
    return data
