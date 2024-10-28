import math

def calculate(T, P, N):
    """
       calculate ITR

       参数:
       N : int : samples number
       P : float : accuracy (0 <= P <= 1)
       T : float : transfer time

       返回:
       float : information transfer rate（ITR）
       """
    itr = (60 / T) * (math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1)))
    return itr