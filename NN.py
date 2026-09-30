import torch as tc
import torch.nn as nn
import copy
import settings as s



class LRP_product(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1,f2):
        self.f1, self.f2 = f1, f2
        self.y = f1*f2
        return self.y


    def relprop(self,R):
        #these are not actually relevances -> only at multiplication at the end
        
        R1 = self.f1 * R/ (self.y + 1e-9 * tc.sign(self.y))
        R1 = tc.nan_to_num(R1, 0.0)
        R2 = self.f2

        return R1.detach(), R2.detach() 


class LRP_Linear(nn.Module):
    def __init__(self, inp, outp, gamma=0.01, eps=1e-5):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.rho = None
        self.iteration = None

    def forward(self, x):

        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device

        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)

        Ap = A.clamp(min=0).detach().data.requires_grad_(True)
        Am = A.clamp(max=0).detach().data.requires_grad_(True)


        zpp = self.newlayer(1).forward(Ap)  
        zmm = self.newlayer(-1, no_bias=True).forward(Am) 

        zmp = self.newlayer(1, no_bias=True).forward(Am) 
        zpm = self.newlayer(-1).forward(Ap) 

        with tc.no_grad():
            Y = self.forward(A).data

        sp = ((Y > 0).float() * R / (zpp + zmm + self.eps * ((zpp + zmm == 0).float() + tc.sign(zpp + zmm)))).data # new version
        sm = ((Y < 0).float() * R / (zmp + zpm + self.eps * ((zmp + zpm == 0).float() + tc.sign(zmp + zpm)))).data

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sm).sum().backward()
        cmp = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sp).sum().backward()
        cmm = Am.grad
        Am.grad = None
        Am.requires_grad_(True)


        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data


        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):

        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=0) # Replace 1e-9 by zero
        else:
            rho = lambda p: p + self.gamma * p.clamp(max=0) # same here

        layer_new = copy.deepcopy(self.linear)

        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass

        return layer_new


class LRP_ReLU(nn.Module):
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R


class LRP_DropOut(nn.Module):
    def __init__(self, p):
        super(LRP_DropOut, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

    def relprop(self, R):
        return R



class LRP_cat(nn.Module):
    def __init__(self):
        super(LRP_cat, self).__init__()

    def forward(self, list_of_tensors):
        self.sizes = [tensor.shape[1] for tensor in list_of_tensors]
        return tc.cat(list_of_tensors, axis=1)

    def relprop(self, R):
        splitted_R = tc.split(R,self.sizes,dim=1)
        return splitted_R


class LRP_DropOut_Linear(nn.Module):
    def __init__(self, inp, outp, p):
        super(LRP_DropOut_Linear, self).__init__()

        self.linear = LRP_Linear(inp, outp)
        self.dropout = LRP_DropOut(p)

    def forward(self, x):
        return self.linear(self.dropout(x))

    def relprop(self, R):
        R = self.linear.relprop(R)
        R = self.dropout.relprop(R)
        return R


class SimpleModel(nn.Module):
    def __init__(self, ds):
        super(Model, self).__init__()

        self.input_features = ds.nfeatures

        self.layers = nn.Sequential(LRP_Linear(self.input_features, 100),
                                     LRP_ReLU(),
                                     LRP_Linear(100,1))




    def forward(self, x):
        return self.layers.forward(x)


    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R
        



class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Model, self).__init__()


        self.layers = nn.Sequential(LRP_DropOut(p=s.input_dropout),
                                     LRP_Linear(n_input, n_hidden),
                                     LRP_ReLU(),
                                     LRP_DropOut(p=s.intermediate_dropout),

                                     LRP_Linear(n_hidden,n_output)

                                     )
        

    def forward(self, x):
        return self.layers.forward(x)


    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R

class Dropout_Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Dropout_Model, self).__init__()

        self.layers = nn.Sequential(LRP_DropOut(p=0.5), #was 0.5!!
                                     LRP_Linear(n_input, n_hidden),
                                     LRP_ReLU(),
                                     LRP_Linear(n_hidden,n_output)

                                     )
        

    def forward(self, x):
        return self.layers.forward(x)


    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R





class Interaction_Model(nn.Module):
    classname = 'Interaction Model'
    def __init__(self, ds):
        super().__init__()

        self.n_therapy, self.n_diagnostics, self.n_product, self.nfeatures_out = ds.n_therapy_features, ds.n_diagnostics_features, s.n_product, ds.n_labels

        self.nn_therapy = LRP_DropOut_Linear(self.n_therapy, s.n_product, p = 0.1)
        self.nn_diagnostics = LRP_DropOut_Linear(self.n_diagnostics, s.n_product, p = 0.5)

        self.last_nn = Model(self.n_product, 1000,self.nfeatures_out)

        self.product = LRP_product()

    def forward(self, therapy, diagnostics):

        intermediate1 = self.nn_therapy(therapy)
        intermediate2 = self.nn_diagnostics(diagnostics)

        if False & self.training:
            if tc.rand(1)<0.5:
                intermediate1 = intermediate1.detach()
            else:
                intermediate2 = intermediate2.detach()

        product = self.product.forward(intermediate1, intermediate2)

        outcome = self.last_nn(product)

        return outcome
    

    def regress(self, therapy, diagnostics):

        intermediate1 = self.nn_therapy(therapy)
        intermediate2 = self.nn_diagnostics(diagnostics)

        if False & self.training:
            if tc.rand(1)<0.5:
                intermediate1 = intermediate1.detach()
            else:
                intermediate2 = intermediate2.detach()


        product = self.product.forward(intermediate1, intermediate2)

        outcome = self.last_nn(product)

        return outcome

    def relprop(self, R):
        device = R.device
        product_relevance = self.last_nn.relprop(R)
        
        factor1_relevance, factor2_relevance = self.product.relprop(product_relevance)


        input_relevance = tc.zeros(R.shape[0], self.n_therapy, self.n_diagnostics, device=device)

        for i in range(self.n_product):

            input1_relevance = self.nn_therapy.relprop(factor1_relevance * tc.eye(self.n_product, device = device)[i])         
            input2_relevance = self.nn_diagnostics.relprop(factor2_relevance * tc.eye(self.n_product, device=device)[i])    

            input_relevance += input1_relevance[:,:,None] * input2_relevance[:,None,:]

        return input_relevance
    






    








class Simple_Model(nn.Module):
    classname = 'Simple Model'
    def __init__(self, ds):
        super().__init__()

        self.n_therapy, self.n_diagnostics, self.n_product, self.nfeatures_out = ds.n_therapy_features, ds.n_diagnostics_features, s.n_product, ds.n_labels

        self.nn = nn.Sequential(LRP_Linear(self.n_therapy + self.n_diagnostics, self.n_product),
                                nn.ReLU() ,
                                LRP_Linear(self.n_product, self.nfeatures_out))

    def forward(self, therapy, diagnostics):

        intermediate = tc.cat((therapy, diagnostics), axis=1)

        outcome = self.nn(intermediate)

        return outcome

    def regress(self, therapy, diagnostics):

        intermediate = tc.cat((therapy, diagnostics), axis=1)

        outcome = self.nn(intermediate)

        return outcome




