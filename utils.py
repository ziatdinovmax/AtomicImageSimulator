import numpy as np

def build_Si_lattice(t0, t1, t2, l1, l2):
    '''Builds a reconstructed Si(100)(2x1) surface'''
    t0 = t0 + t0*((np.random.randint(0, 30)-20)/100)
    t1 = t1 + t1*(np.random.randint(0, 15)/100)
    t2 = t2 + t2*(np.random.randint(0, 50)/100)
    dimer = np.array([[0., 0.], [0., t0]])
    dimer_row = np.empty((0, 2))
    for i in range(l1):
        dimer_ = np.copy(dimer)
        dimer_[:, 0] = dimer_[:, 0] + t1*(i+1)
        dimer_row = np.append(dimer_row, dimer_, axis=0)
        all_rows = np.empty((0, 2))
    for i in range (l2):
        dimer_row_ = np.copy(dimer_row)
        dimer_row_[:, 1] = dimer_row_[:, 1] + t2*(i+1)
        all_rows = np.append(all_rows, dimer_row_, axis=0)
    labels = np.array(['Si']*len(all_rows))
    all_rows = np.concatenate((labels[:, None], all_rows), axis=1)
    return all_rows

def check_batch(batch, ch=1):
    '''
    Checks if there an "impurity" channel is not empty
    for every image in a batch
    '''
    vals = []
    for ar in [np.unique(y) for y in batch[:,ch,:, :]]:
        if len(ar) == 1:
            vals.append(1)
        else:
            vals.append(0)
    return vals

def squeeze_channels(y_train):
    '''Squeezes multiple channel into a single channel'''
    y_train_ = np.zeros((y_train.shape[0], y_train.shape[2], y_train.shape[3]))
    for c in range(y_train.shape[1]):
        y_train_ += y_train[:, c]*c
    y_train[y_train<0] = 0
    return y_train_