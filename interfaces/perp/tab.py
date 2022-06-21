import numpy as np
import numpy.ma as ma
from numba import njit
import pickle


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

    def to_ndarray(self, save_npy=False):
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
                # print(i, field)
                # print(np.any(data == 0))
                data_ = np.reshape(data_, (nP, nT)).T

            if save_npy:
                print('\nSaving table to: ' + self.tab['title'] + '.npy\n')
                template = 'table_{}_{}.npy'
                np.save(template.format(self.tab['title'], field), data_)
            else:
                self.data.append(data_)

        if save_npy:
            fname = 'stats_{}.pkl'.format(self.tab['title'])
            with open(fname, 'wb') as f:
                pickle.dump(self.tab, f)

        else:
            return self.data, self.tab

    def plot(self, i_field):
        # implement later
        fld = self.tab["fields"][i_field]
        return None

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

    def remove_nans(self, original):
        new = np.array(original)
        for where in np.argwhere(np.isnan(original)):
            i, j = where
            new[i, j] = self._interp_nans(original, i, j)
        return new
