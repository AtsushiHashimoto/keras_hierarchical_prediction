import warnings
import keras.backend as K
class PerGroupFunc():
    def __init__(self, is_sequential=False):
        self.funcs = []
        self.scopes = [] # (start,end) tuples
        self.weights = []
        self.is_sequential=is_sequential
        
    def register(self, func, scope, weight=1.0):
        self.funcs.append(func)
        self.scopes.append(scope)
        self.weights.append(weight)
        
    @staticmethod
    def gen_slice(scope, len_shape):
        return [slice(None)]*(len_shape-1) + [slice(*scope)]
        
        
    def call(self, y_true, y_pred):
        if len(self.funcs)==0:
            warnings.warn("No loss functions are set.")
            return 0
        len_shape = len(y_true.shape)
        # value and validity(typically, 0.f or 1.f)        
        val_validity = [f(y_true[self.gen_slice(sc,len_shape)], y_pred[self.gen_slice(sc,len_shape)]) for (f,sc) in zip(self.funcs, self.scopes)]            
                     
        is_first = True
        for vs, w in zip(val_validity, self.weights):
            if is_first:
                is_first = False
                val = vs[0]*w
                _sum = vs[1]
                continue
            val += vs[0]*w
            _sum += vs[1]
        return val / _sum
