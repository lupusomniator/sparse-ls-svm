import numpy as np
import progbar
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import (
    chi2_kernel,
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)
def ScalarProduct(x, y, args=None):      
    return x @ y

def Polynomial(x, y, args=None):
    if args is None:
        args = {'c': 1, 'd': 1}
    return (x @ y + args['c'])**args['d'] 

def Gaussian(x, y, args=None):
    if args is None:
        args = {'sigma_squared': 1}
    dif = x-y
    return np.e**(-(dif @ dif)/(2 * args['sigma_squared']))

def Laplacian(x, y, args=None): 
    if args is None:
        args = {'alpha': 1}
    dif = x-y
    return np.e**(-args['alpha'] * np.sqrt(dif @ dif))

class TMultikernelLSSVM:
    '''
    Class for LS-SVM
    '''

    available_kernels = {
        'linear':linear_kernel,
        'poly': polynomial_kernel,
        'rbf':rbf_kernel,
        'chi2':chi2_kernel
    }

    def __init__(self, kernels: list = [{'name':'rbf', 'params': {'gamma': 0.5}}], C: float = 0.5):  
        '''
        Class initializer
        :param kernels: used kernels
             "kernels" format:
             list(
             {
               'name': _kernel_name_ <- string
               'kernel': _kernel_function <- function. Not obligatory, is determined automatically.
               'params': _kernel_parameters <- dictionary. Individual for each kernel
             },...
            )
        :param C: initial restriction coefficient
        '''
        self.X = None
        self.y_train = None
        self.kernels = None
        self.K = []  # Matrix of kernels, normed
        self.diags = []  # Diagonals of source matrix of kernels
        self.lambd = 0  # Coefficients of algorithm
        self.omega0 = 0  # Algorithm offset
        self.C = C  # Restriction coefficient
        self.__set_kernels__(kernels)

    def __set_kern_coef__(self, kern_coef:list):
        '''
        Setting kernel coefficients
        :param kern_coef: kernel coefficients for model
        '''
        if len(kern_coef)!=self._kernel_n:
            return
        self.kern_coef = [0 for _ in range(self._kernel_n)]
        for i in range(len(kern_coef)):
            self.kern_coef[i] = kern_coef[i]

    def __set_kernels__(self, kernels: list):   
        '''
        Setting kernels for model
        :param kernels: used kernels
        '''
        self.kernels = []
        for i in range(len(kernels)):            
            for el in list(kernels[i].keys()):
                if not (el == 'name' or el == 'params'):
                    del kernels[i][el]
            if 'name' in kernels[i]:
                if kernels[i]['name'] in self.available_kernels:                    
                    kernels[i]['func'] = self.available_kernels[kernels[i]['name']]
                else: 
                    continue
                if not 'params' in kernels[i]:
                    kernels[i]['params'] = None
                self.kernels.append(kernels[i])
        self._kernel_n = len(self.kernels)
        self.__set_kern_coef__([1.0 / self._kernel_n for _ in range(self._kernel_n)])
        self.kern_matr_formed = False

    def __form_kern_matrix__(self, X, y, progress = False): 
        '''
        The formation of the matrix of kernels
        :param X: train data, used in model
        :param y: train labels, used in model
        :param progress: show or do not show the progress
        '''
        n = len(y)
        # self.K = []
        # self.diags = []
        # for k in range(self._kernel_n):
        #     kern = self.kernels[k]
        #     self.diags.append(list(map(lambda x: kern['func'](x, x, **kern['params']),X)))
        # self.diags = np.array(self.diags)
        # p = 1  
        self.K = np.zeros((self._kernel_n, n, n))
        for d in range(self._kernel_n):
            kernel = self.kernels[d]
            self.K[d] = kernel['func'](X, X, **kernel['params'])
            # diag = self.diags[d]
            # for i in range(n):
            #     for j in range(i, n):
            #         self.K[d][i][j]  = self.K[d][j][i] = kernel['func'](X[i], X[j], **kernel['params']) / np.sqrt(diag[i] * diag[j])
                
            #     if progress: 
            #         progbar.set_progress(p / float(n * self._kernel_n)) 
            #     p += 1
        self.kern_matr_formed = True
    
    def __set_train_data__(self, X_train, y_train, kern_matrix = None, kern_diag = None, progress = False):
        '''
        Setting train data for model. Usefull for cross validation.
        :param X_train: train data features
        :param y_train: train data labels
        '''
        try:
            self.X = X_train.values #matrix, nxm
        except Exception:
            self.X = np.array(X_train) #matrix, nxm
            
        try:
            self.y = y_train.values #matrix, nxm
        except Exception:
            self.y = np.array(y_train) #matrix, nxm
        # Forming the matrix of kernels. This can increase the speed of calculations
        if kern_matrix is None or kern_diag is None:
            self.__form_kern_matrix__(self.X, self.y)
        else:
            self.K = kern_matrix
            self.diags = kern_diag
            
    def fit(self, X_train = None, y_train = None, train_index = None):
        '''
        Training the model
        :param X_train: train data features. None, if data was already setted.
        :param y_train: train data labels. None, if data was already setted.
        :param train_index: indices in train data used for model training. None, if want to use all data.
        :param progress: show or do not show the progress
        '''
            
        def __choose_data(train_index):
            '''
            Selecting items from train set with special indices
            :param train_index: indices that you want to select
            '''
            if train_index is None:
                return self.X, self.y, self.K, self.diags
            if self.kern_matr_formed:
                return  self.X[train_index],\
                        self.y[train_index],\
                        np.array([[[val for val in line[train_index]] for line in kernel[train_index]] for kernel in self.K]),\
                        np.array([kernel[train_index] for kernel in self.diags])
            else:
                return  self.X[train_index],\
                        self.y[train_index],\
                        None,\
                        np.array([kernel[train_index] for kernel in self.diags])
                        
        if not X_train is None:
            if not y_train is None:
                self.__set_train_data__(X_train, y_train)
            else:
                print("X_train and y_train should be either both None, either both not None")
        else:
            if not y_train is None:
                print("X_train and y_train should be either both None, either both not None")
        if train_index is None:
            train_index = np.array(range(len(self.y)))
        self.last_train_X, self.last_train_y, self.last_train_K, self.last_train_diags = (
            __choose_data(train_index)
        )
        X = self.last_train_X
        y = self.last_train_y
        K = self.last_train_K
        n = len(y)
        
        # Omega = (ZZ^t+I/C) - part of algorithm
        # Z = [x1^t*y1, ..., xn^t*yn]
        Omega = np.zeros((n, n), dtype=float)
        
        # Larning
        if self.kern_matr_formed:
            for i in range(n):
                for j in range(i,n):
                    el=0
                    for k in range(self._kernel_n):
                        el += self.kern_coef[k] * K[k][i][j] * y[i] * y[j]    
                    Omega[i, j], Omega[j, i] = el, el
                Omega[i, i] += 1.0 / self.C
        else:
            for i in range(n):
                for j in range(i,n):
                    el=0.0
                    for k in range(self._kernel_n):
                        kernel = self.kernels[k]
                        diag = self.diags[k]
                        el +=   self.kern_coef[k] * kernel['func'](X[i],X[j],kernel['params'])/np.sqrt(diag[i] * diag[j]) * y[i] * y[j]
                                                      
                    Omega[i, j], Omega[j, i] = el, el
                Omega[i, i] += 1.0 / self.C
               
        Omega = np.linalg.inv(Omega)
        ones = np.ones(n)
        
        self.omega0 = (y.T @ Omega @ ones)/(y.T @ Omega @ y)
        self.lambd = Omega @ (ones - y * self.omega0)
        
        return self.lambd, self.omega0
    
    
    def find_best_C(self, range_C: (int,int) = (1,100000), precision = 2,
                    kf_n_splits:int = 4, kf_random_state = None,
                    X_train = None, y_train = None):
        '''
        Finding best restriction coefficient for model using cross validation based on Accuracy
        :param range_C: range of values in which the search will occur
        :param X_train: train data features. None, if data was already setted.
        :param y_train: train data labels. None, if data was already setted.
        '''
        save_C = self.C
        def do_cv_avg(C):
            self.C = C
            avg_acc = 0
            for train_index, test_index in kf.split(self.X):
                self.fit(train_index = train_index)
                y_pred = self.predict(self.X[test_index])
                avg_acc += metrics.accuracy_score(self.y[test_index], y_pred)
            return avg_acc/kf_n_splits
            
        left_bound = min(range_C)
        if left_bound<1:
            left_bound = 1
        right_bound = max(range_C)
        if right_bound<1:
            right_bound = 1
        reduction = right_bound - left_bound
        kf = KFold(n_splits = kf_n_splits, shuffle = True, random_state = kf_random_state)
        
        golden_rate = (3 - 5**0.5)/2
        x1 = left_bound + (right_bound - left_bound) * golden_rate
        x2 = right_bound - (right_bound - left_bound) * golden_rate
        
        y1 = do_cv_avg(x1)
        y2 = do_cv_avg(x2)
        
        report = {x1:y1, x2:y2}
        print(reduction)
        while reduction > precision:
            if y1 < y2:
                right_bound = x2
                x2 = x1
                x1 = left_bound + (right_bound - left_bound) * golden_rate
                y1 = do_cv_avg(x1)
            else:
                left_bound = x1
                x1 = x2
                x2 = right_bound - (right_bound - left_bound) * golden_rate
                y2 = do_cv_avg(x2)
                
            report[x1] = y1
            report[x2] = y2
            reduction = right_bound - left_bound
            
            print(reduction)
        self.C = save_C
        return  x1 if y1 > y2 else x2,\
                y1 if y1 > y2 else y2,\
                report
        
    def predict_single(self, x):
        '''
        Single item classification
        :param x: item needed to be classified
        '''
        _sum = 0.0
        n = len(self.last_train_y)
        for k, kernel in enumerate(self.kernels):
            diagx = kernel['func'](x.reshape(1, -1), x.reshape(1, -1), **kernel['params'])
            kparam = kernel['params']
            kfunc = kernel['func']
            kern_values = self.kern_coef[k] * kfunc(self.last_train_X, x.reshape(1, -1), **kparam)
            for i in range(n):                
                _sum += self.lambd[i] * kern_values[i] * self.last_train_y[i]
        return np.sign( _sum + self.omega0 )[0]
    
     
    def predict(self, X):
        '''
        Classification of dataset
        :param X: items needed to be classified
        '''
        try:
            X = X.values  # matrix, nxm
        except Exception:
            X = np.array(X)  # matrix, nxm
        return list(map(self.predict_single, X))
