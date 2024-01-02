import math

def compute_parallel_ratio(m1, m2): 
    """
    Given two slopes m1 and m2, return a 
    ratio value of how parallel the two lines are

    Argument 
    ----- 

    m1 : int
    m2 : int 

    Return 
    ----- 

    parallel_ratio : float
        ratio of how parallel these two lines are
    """

    parallel_ratio = abs((m1 - m2) / (m1 + m2) / 2)
    return parallel_ratio