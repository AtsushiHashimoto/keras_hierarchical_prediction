import warnings
import keras.backend as K
class MergeLossGenerator():
    def __init__(self):
        self.loss_funcs = []
        self.scopes = [] # (start,end) tuples
        self.weights = []
    def register(self, func, scope, weight=1.0):
        self.loss_funcs.append(func)
        self.scopes.append(scope)
        self.weights.append(weight)
    def loss(self, y_true, y_pred):
        if len(self.loss_funcs)==0:
            warnings.warn("No loss functions are set.")
            return 0
        losses = [w*f(y_true[:,s:e],y_pred[:,s:e]) for (f,(s,e),w) in zip(self.loss_funcs,self.scopes,self.weights)]
        loss = K.sum(losses[0],axis=-1,keepdims=False)
        for l in losses[1:]:
            loss = loss + K.sum(l, axis=-1,keepdims=False)
        return loss
