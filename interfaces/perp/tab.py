import numpy as np
import numpy.ma as ma
from numba import njit
import pickle

def _interp_nans(arr, x, y):
    nr, nc = arr.shape
    # print(x, y)
    # define square around x, y coordinates in array
    dist = 1
    x1, x2 = max(0, x - dist), min(nr, x + 1 + dist)
    y1, y2 = max(0, y - dist), min(nc, y + 1 + dist)
    square = np.array(arr[x1: x2, y1:y2])

    # if all elements in this square are NaNs, make it larger
    while np.all(np.isnan(square)):
        # print('too small')
        dist += 1
        x1, x2 = max(0, x - dist), min(nr, x + 1 + dist)
        y1, y2 = max(0, y - dist), min(nc, y + 1 + dist)
        square = np.array(arr[x1: x2, y1:y2])

    square = ma.masked_array(square, mask=np.isnan(square))
    return square.mean()


class Tab:
    def __init__(self, inpfl):
        self.inpfl = inpfl
        self.tab = {}
        with open(inpfl, 'r') as fl:
            _ = fl.readline().strip()  # skip line..
            title = fl.readline().strip()  # remove blanks...
            _ = fl.readline().strip()
            _ = fl.readline().strip()
            _ = fl.readline().strip()
            dT = float(fl.readline())
            nT = float(fl.readline())
            _ = fl.readline().strip()
            _ = fl.readline().strip()
            dP = float(fl.readline())
            nP = float(fl.readline())
            nvar = float(fl.readline())
            variables = fl.readline()

        nT, nP, nvar = int(nT), int(nP), int(nvar)

        # read actual data
        print('\nReading table:  ' + title)

        fields = variables.split()
        self.tab['title'] = ''.join(title.split(".")[:-1])
        self.tab['dT'] = dT
        self.tab['dP'] = dP
        self.tab['nT'] = nT
        self.tab['nP'] = nP
        self.tab['nvar'] = nvar
        self.tab['fields'] = fields

        print('dT: ', dT, ', dP: ', dP, ', nT: ',
              nT, ', nP: ', nP, ', nvar: ', nvar)
        self.data = []  # could parallelize method, but need 2 initialize data

    def load(self):
        nT = self.tab['nT']
        nP = self.tab['nP']
        for i, field in enumerate(self.tab['fields']):
            # print(i, field)
            data_ = np.loadtxt(self.inpfl, skiprows=13, usecols=i)
            if field == 'T(K)':
                data_ = data_[:nT]

            elif field == 'P(bar)':
                data_ = data_[::nT]

            else:
                print(i, field)
                # print(np.any(data == 0))
                data_ = np.reshape(data_, (nP, nT)).T
                
            self.data.append(data_)
        
    def save(self):
        for i, field in enumerate(self.tab['fields']):
            data_ = self.data[i]
            print('\nSaving table to: ' + self.tab['title'] + '.npy\n')
            template = 'table_{}_{}.npy'
            np.save(template.format(self.tab['title'], field), data_)
    
        fname = 'stats_{}.pkl'.format(self.tab['title'])
        with open(fname, 'wb') as f:
            pickle.dump(self.tab, f)
        

    def plot(self, i_field):
        # implement later
        fld = self.tab["fields"][i_field]
        return None
    
    def remove_nans(self):
        for k, (f, n) in enumerate(zip(self.data, self.tab["fields"])):
            if n != "T(K)" and n != "P(bar)":
                new = np.array(f)
                for where in np.argwhere(np.isnan(f)):
                    i, j = where
                    new[i, j] = _interp_nans(f, i, j)
                self.data[k] = new
