import numpy as np
from scipy.spatial.distance import cdist, cosine
from math import isinf
import scipy.io.wavfile as scwav
import librosa
import pylab

def constrained_traceback(backpointer_mat, D):
    """
    Computes the optimal path for warping.

    :param array backpointer_mat: N1*N2 array
    :param array D: accumulated cost matrix
    Returns the warp path cordinates.
    """
    cords = []
#    if D.shape[0] > D.shape[1]:
#        min_idx = np.argmin(D[:,-1])
#        cords = [[min_idx, D.shape[1]-1]]
#    else:
#        min_idx = np.argmin(D[-1,:])
#        cords = [[D.shape[0]-1, min_idx]]
    
    cords = [[D.shape[0]-1, D.shape[1]-1]]
    
    if min(cords[-1]) > 0:
        can_traceback = True
    
    while can_traceback:
        cand_row = int(backpointer_mat[cords[-1][0],cords[-1][1],0])
        cand_col = int(backpointer_mat[cords[-1][0],cords[-1][1],1])
        cords.append([cand_row, cand_col])
        
        if (cand_col==0) or (cand_col==0):
            can_traceback = False

    return np.asarray(cords)

def constrained_dtw(x, y, dist=cosine):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the cost matrix, the accumulated cost matrix, and the warp path.
    """
    assert len(x)>0
    assert len(y)>0

    r, c = len(x), len(y)
    D_global = np.zeros((r + 2, c + 1))
    D_global[0:2, :] = np.inf
    D_global[:, 0] = np.inf
    D_global[1,0] = 0
    backpointer = np.zeros((r,c), dtype=(float,2))
#    D_local = D_global[2:, 1:]  # view

#    D_local = cdist(x, y, dist)
    D_global[2:,1:] = cdist(x, y, dist)
    C = D_global[2:,1:].copy()
    
    for i in range(2,r+2):
        for j in range(1,c+1):
            rows_considered = [i, i-1, i-2]
            cols_considered = [j-1, j-1, j-1]
            
            min_list = [D_global[i, j-1]]
            min_list += [D_global[i-1,j-1], D_global[i-2,j-1]]
            D_global[i, j] += min(min_list)
            
            min_idx = np.argmin(min_list)
            
            backpointer[i-2,j-1,0] = max(0, rows_considered[min_idx] - 2)
            backpointer[i-2,j-1,1] = max(0, cols_considered[min_idx] - 1)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = constrained_traceback(backpointer, D_global[2:,1:])
    return C, D_global[2:,1:], path

if __name__ == '__main__':

    s = scwav.read('./data/evaluation/neutral_5/1010.wav')
    s = np.asarray(s[1], np.float64)
    t = scwav.read('./data/evaluation/angry_5/1010.wav')
    t = np.asarray(t[1], np.float64)
    
    s_mfc = librosa.feature.mfcc(y=s, sr=16000, \
            hop_length=int(16000*0.005), win_length=int(16000*0.005), \
            n_fft=1024, n_mels=128)
    
    t_mfc = librosa.feature.mfcc(y=t, sr=16000, \
            hop_length=int(16000*0.005), win_length=int(16000*0.005), \
            n_fft=1024, n_mels=128)

    cost, acc_cost, path = constrained_dtw(s_mfc.T, t_mfc.T, cosine)
    
    pylab.imshow(cost)
    path = np.asarray(path)
    pylab.plot(path[:,1], path[:,0], 'r-')
    pylab.figure(), pylab.imshow(acc_cost), pylab.plot(path[:,1], path[:,0], 'r-')

    # Vizualize
#    from matplotlib import pyplot as plt
#    plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
#    plt.plot(path[0], path[1], '-o')  # relation
#    plt.xticks(range(len(x)), x)
#    plt.yticks(range(len(y)), y)
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.axis('tight')
#    if isinf(w):
#        plt.title('Minimum distance: {}, slope weight: {}'.format(dist, s))
#    else:
#        plt.title('Minimum distance: {}, window widht: {}, slope weight: {}'.format(dist, w, s))
#    plt.show()