import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from core.wing import FAN, HighPass
from   numpy.random            import RandomState
from scipy.stats import chi
from torch.nn import Module, init
from QGAN.utils.quaternion_layers import QuaternionConv, QuaternionLinear
from htorch import quaternion, layers, utils

f = open('config.json',)

# returns JSON object as
# a dictionary
data = json.load(f)
args = Munch(data)

args["layer_norm"] = False if args["layer_norm"] =="False" else True
args["quat_inst_norm"]= False if args["quat_inst_norm"] =="False" else True
args["quat_max_pool"]= False if args["quat_max_pool"] =="False" else True
args["qsngan_layers"] = False if args["qsngan_layers"] =="False" else True
args["htorch_layers"]= False if args["htorch_layers"] =="False" else True
args["real"]= False if args["real"] =="False" else True
args["phm"]= False if args["phm"] =="False" else True
args["last_dense"]= False if args["last_dense"] =="False" else True
device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')

class QuaternionInstanceNorm2d(nn.Module):
    r"""Applies a 2D Quaternion Instance Normalization to the incoming data.
        """

    def __init__(self, N, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(QuaternionInstanceNorm2d, self).__init__()
        self.num_features = num_features // N
        self.gamma_init = 1.
        self.affine = affine
        self.N = N
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * self.N, 1, 1), requires_grad=self.affine)
        self.eps = torch.tensor(1e-5)
        #TODO remove this
        # self.register_buffer('moving_var', torch.ones(1) )
        # self.register_buffer('moving_mean', torch.zeros(4))
        ####
        self.momentum = momentum
        self.track_running_stats = track_running_stats

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.affine)

    def forward(self, input):
        # print(self.training)
        new_input = self.do_chunks(input)
        return new_input


    def do_chunks(self,input):
        n = self.N
        quat_components = torch.chunk(input, n, dim=1)
        mu_list = []

        #r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        for unit in quat_components:
            mu_ = torch.mean(unit, axis=(2,3), keepdims=True)
            mu_list.append(torch.mean(mu_))
        
        mu = torch.stack(mu_list, dim=0)
        
        #mu = torch.cat([mu_r,mu_i, mu_j, mu_k], dim=1)

        delta_l = []
        qvar = 0
        for unit,mu in zip(quat_components,mu_list):
            delta_x = unit - mu
            delta_l.append(delta_x)
            qvar+=delta_x**2

        
        quat_variance = torch.mean(qvar)

        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        norm_list = []
        for d in delta_l:
            norm_list.append(d/denominator)
        
        beta_components = torch.chunk(self.beta, n, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_inp = []
        
        for i,norm in enumerate(norm_list):
            new_inp.append((self.gamma*norm)+beta_components[i])

        return torch.cat(new_inp, dim=1)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N=' + str(self.N) \
               + ', num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'



def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1,1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)

def get_weight(n,in_f,out_f,kernel_size,criterion):
    r, i, j, k = quaternion_init(
        in_f,
        out_f//n,
        rng=RandomState(777),
        kernel_size=kernel_size,
        criterion=criterion
    )
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    if criterion=="he":
        return torch.cat([r,i,j,k],dim=1)
    else:
        return torch.cat([r.squeeze(1),i.squeeze(1),j.squeeze(1),k.squeeze(1)],dim=0)

def get_s_init(kernel_size,in_f,out,criterion):
    if criterion == "glorot":
        w_shape = (out, in_f) + (*kernel_size,)
        r_weight = torch.Tensor(*w_shape)
        i_weight = torch.Tensor(*w_shape)
        j_weight = torch.Tensor(*w_shape)
        k_weight = torch.Tensor(*w_shape)
    else:
        r_weight = torch.Tensor(in_f,out)
        i_weight = torch.Tensor(in_f,out)
        j_weight = torch.Tensor(in_f,out)
        k_weight = torch.Tensor(in_f,out)

    r, i, j, k = quaternion_init(
        r_weight.size(1),
        r_weight.size(0),
        rng=RandomState(args.seed),
        kernel_size=kernel_size,
        criterion=criterion
    )

    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)
    
    return torch.stack([r_weight,i_weight,j_weight,k_weight],dim=0).to(device)

########################
## STANDARD PHM LAYER ##
########################

class PHMLinear(nn.Module):

  def __init__(self, n, in_features, out_features,bias=True):
    super(PHMLinear, self).__init__()
    self.n = n #if ((in_features //n>0) and (out_features //n>0) ) else 1
    self.in_features = in_features
    self.out_features = out_features

    # self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.n, self.n))))
    # self.s = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.out_features//self.n, self.in_features//self.n))))
    #self.a = get_a_init(self.n,None,"he")
    if in_features % n != 0 or out_features % n != 0:
        raise Exception(str(in_features), str(out_features))
    # if self.n == 4:
    #     mat1 = torch.eye(4).view(1, 4, 4)

    #     # Define the four matrices that summed up build the Hamilton product rule.
    #     mat2 = torch.tensor([[0, -1, 0, 0],
    #                         [1, 0, 0, 0],
    #                         [0, 0, 0, -1],
    #                         [0, 0, 1, 0]]).view(1, 4, 4)
    #     mat3 = torch.tensor([[0, 0, -1, 0],
    #                         [0, 0, 0, 1],
    #                         [1, 0, 0, 0],
    #                         [0, -1, 0, 0]]).view(1, 4, 4)
    #     mat4 = torch.tensor([[0, 0, 0, -1],
    #                         [0, 0, -1, 0],
    #                         [0, 1, 0, 0],
    #                         [1, 0, 0, 0]]).view(1, 4, 4)
    #     self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    #     self.s = nn.Parameter(get_s_init(None, self.in_features//self.n,self.out_features//self.n,"he"))
    self.a = nn.Parameter(torch.randint(low=-1,high=2,size=(n,n,n)).type(dtype=torch.FloatTensor))
    self.s = nn.Parameter(get_s_init(None, self.in_features//self.n,self.out_features//self.n,"he")[:self.n,:,:])

    self.old_weight = torch.zeros((self.out_features, self.in_features))
    self.weight = get_weight(self.n,self.out_features,self.in_features,None,"he")
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_features))
        init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter('bias', None)

    #self.reset_parameters()
  def kronecker_product1(self, a, b): #adapted from Bayer Research's implementation
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features))
    for i in range(self.n):
        H = H + torch.kron(self.a[i], self.s[i])
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
    #self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
    return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init(None, self.in_features//self.n,self.out_features//self.n,"he"))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if self.bias is not None:
        init.uniform_(self.bias, -bound, bound)

#############################
## CONVOLUTIONAL PHM LAYER ##
#############################

class PHMConv(nn.Module):

  def __init__(self, n, in_features, out_features, kernel_size, stride=1, padding=0,bias = True):
    super(PHMConv, self).__init__()
    self.n = n #if ((in_features //n>0) and (out_features //n>0) ) else 1
    self.in_features = in_features# if in_features //n>0 else n
    self.out_features = out_features# if out_features //n>0 else n
    if in_features % n != 0 or out_features % n != 0:
        raise Exception(str(in_features), str(out_features))
    self.padding = padding
    self.stride = stride
    # self.a = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.n, self.n))))
    # self.s = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((self.n, self.out_features//self.n, self.in_features//self.n, kernel_size, kernel_size))))
    self.kernel_size = kernel_size
    #self.a = nn.Parameter(get_a_init(self.n,(kernel_size,kernel_size),"glorot"))
    # if self.n == 4:
    #     mat1 = torch.eye(4).view(1, 4, 4)

    #     # Define the four matrices that summed up build the Hamilton product rule.
    #     mat2 = torch.tensor([[0, -1, 0, 0],
    #                         [1, 0, 0, 0],
    #                         [0, 0, 0, -1],
    #                         [0, 0, 1, 0]]).view(1, 4, 4)
    #     mat3 = torch.tensor([[0, 0, -1, 0],
    #                         [0, 0, 0, 1],
    #                         [1, 0, 0, 0],
    #                         [0, -1, 0, 0]]).view(1, 4, 4)
    #     mat4 = torch.tensor([[0, 0, 0, -1],
    #                         [0, 0, -1, 0],
    #                         [0, 1, 0, 0],
    #                         [1, 0, 0, 0]]).view(1, 4, 4)
    #     self.a = nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    #     self.s = nn.Parameter(get_s_init((kernel_size,kernel_size),self.in_features//self.n,self.out_features//self.n,"glorot"))

    # if self.n == 3:
    self.a = nn.Parameter(torch.randint(low=-1,high=2,size=(n,n,n)).type(dtype=torch.FloatTensor) )
    self.s = nn.Parameter(get_s_init((kernel_size,kernel_size),self.in_features//self.n,self.out_features//self.n,"glorot")[:self.n,:,:,:])

    self.weight = torch.zeros((self.out_features, self.in_features))
    #self.weight = get_weight(self.n,1,self.out_features,self.kernel_size,"glorot")
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if bias:
        self.bias = nn.Parameter(torch.Tensor(self.out_features))
        init.uniform_(self.bias, -bound, bound)
    else:
        self.register_parameter('bias', None)
    #self.reset_parameters()

  def kronecker_product1(self, a, s):
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(s.shape[-4:-2]))
    siz2 = torch.Size(torch.tensor(s.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3).unsqueeze(-1).unsqueeze(-1) * s.unsqueeze(-4).unsqueeze(-6)
    siz0 = res.shape[:1]
    out = res.reshape(siz0 + siz1 + siz2)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features, self.kernel_size, self.kernel_size)).to(device)
    for i in range(self.n):
        kron_prod = torch.kron(self.a[i], self.s[i]).view(self.out_features, self.in_features, self.kernel_size, self.kernel_size).to(device)
        H = H + kron_prod
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.a, self.s), dim=0)
    # self.weight = self.kronecker_product2()
    input = input.type(dtype=self.weight.type())
    return F.conv2d(input, weight=self.weight, stride=self.stride, padding=self.padding,bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    # init.kaiming_uniform_(self.a, a=math.sqrt(5))
    # init.kaiming_uniform_(self.s, a=math.sqrt(5))
    mat1 = torch.eye(4).view(1, 4, 4)

    # Define the four matrices that summed up build the Hamilton product rule.
    mat2 = torch.tensor([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 0, 1, 0]]).view(1, 4, 4)
    mat3 = torch.tensor([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]]).view(1, 4, 4)
    mat4 = torch.tensor([[0, 0, 0, -1],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]]).view(1, 4, 4)
    self.a=nn.Parameter(torch.cat([mat1,mat2,mat3,mat4],dim=0))
    self.s = nn.Parameter(get_s_init((self.kernel_size,self.kernel_size),self.in_features//self.n,self.out_features//self.n,"glorot"))

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    if self.bias is not None:
        init.uniform_(self.bias, -bound, bound)



fst,snd = False,False
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        if args.qsngan_layers:
            self.qconv1=QuaternionConv(dim_in, dim_in, kernel_size=3, stride=1, padding=1)
            self.qconv2=QuaternionConv(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        elif args.phm:
            self.qconv1=PHMConv(args.N,dim_in, dim_in, kernel_size=3, stride=1, padding=1)
            self.qconv2=PHMConv(args.N,dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        elif args.htorch_layers:
            self.qconv1=layers.QConv2d(dim_in//4, dim_in//4, kernel_size=3, stride=1, padding=1)
            self.qconv2=layers.QConv2d(dim_in//4, dim_out//4, kernel_size=3, stride=1, padding=1)
        elif args.real:
            self.qconv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
            self.qconv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        
        if args.quat_max_pool:
            self.maxpool = nn.MaxPool2d(2, 2, 0, return_indices=True)

        if self.normalize:
            if args.layer_norm:
                global fst,snd
                diff = dim_out-dim_in 
                if diff==128:
                    self.norm1 = nn.LayerNorm([dim_in,dim_in,dim_in], elementwise_affine=True)
                    self.norm2 = nn.LayerNorm([dim_in,dim_in//2,dim_in//2], elementwise_affine=True)
                if diff == 256:
                    self.norm1 = nn.LayerNorm([dim_in,dim_in//4,dim_in//4], elementwise_affine=True)
                    self.norm2 = nn.LayerNorm([dim_in,dim_in//8,dim_in//8], elementwise_affine=True)
                if diff == 0:
                    if not fst:
                        self.norm1 = nn.LayerNorm([dim_in,dim_in//16,dim_in//16], elementwise_affine=True)
                        self.norm2 = nn.LayerNorm([dim_in,dim_in//32,dim_in//32], elementwise_affine=True)
                        fst = True
                    elif not snd:
                        self.norm1 = nn.LayerNorm([dim_in,dim_in//32,dim_in//32], elementwise_affine=True)
                        self.norm2 = nn.LayerNorm([dim_in,dim_in//64,dim_in//64], elementwise_affine=True)
                        snd = True
                    else:
                        self.norm1 = nn.LayerNorm([dim_in,dim_in//64,dim_in//64], elementwise_affine=True)
                        self.norm2 = nn.LayerNorm([dim_in,dim_in//64,dim_in//64], elementwise_affine=True)
            elif args.quat_inst_norm:
                self.norm1 = QuaternionInstanceNorm2d(args.N, dim_in, affine=True)
                self.norm2 = QuaternionInstanceNorm2d(args.N, dim_in, affine=True)
            else:
                self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
                self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)

        if self.learned_sc:
            if args.qsngan_layers:
                self.qconv1x1=QuaternionConv(dim_in, dim_out,  kernel_size=1, stride=1, padding=0, bias = False)
            elif args.phm:
                self.qconv1x1=PHMConv(args.N,dim_in, dim_out,  kernel_size=1, stride=1, padding=0, bias = False)
            elif args.htorch_layers:
                self.qconv1x1=layers.QConv2d(dim_in//4, dim_out//4, kernel_size=1, stride=1, padding=0, bias = False)
            elif args.real:
                self.qconv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def q_max_pool2d(self,x):
        c, idx = self.maxpool(x.norm().to(device))
        idx = torch.cat([idx]*4, 1).to(device)
        flat = x.torch().flatten(start_dim=2).to(device)
        output = flat.gather(dim=2, index=idx.flatten(start_dim=2)).view_as(idx).to(device)
        return output
    
    def _shortcut(self, x):
        if self.learned_sc:
            # print(x.shape,"learned")
            x = self.qconv1x1(x)
            #x = self.conv1x1(x)
            # print(x.shape,"learned")

        if self.downsample:
            if args.quat_max_pool:
                x = self.q_max_pool2d(x)
            else:
                x = F.avg_pool2d(x.torch(),2) if isinstance(x,quaternion.QuaternionTensor) else F.avg_pool2d(x,2)
            # print(x.shape,"down")
            #x = F.avg_pool2d(x, 2)
        
        return x

    def _residual(self, x):
        if self.normalize:
            try:
                if isinstance(x,quaternion.QuaternionTensor):
                    x = self.norm1(x.torch()).to(device)
                else:
                    x = self.norm1(x).to(device)
            except:
                print("norm1",self.norm1,x.size()[1:])
                self.norm1 = nn.LayerNorm(x.size()[1:]).to(device)
                x = self.norm1(x).to(device)

        x = self.actv(x)
        x = self.qconv1(x)
        #print(x.shape, "ee")
        if self.downsample:
            if args.quat_max_pool:
                x = self.q_max_pool2d(x)
            else:
                x = F.avg_pool2d(x.torch(),2) if isinstance(x,quaternion.QuaternionTensor) else F.avg_pool2d(x,2)
        if self.normalize:
            try:
                if isinstance(x,quaternion.QuaternionTensor):
                    x = self.norm2(x.torch()).to(device)
                else:                            
                    x = self.norm2(x).to(device)
            except:
                print("norm2",self.norm2,x.size()[1:])
                self.norm2 = nn.LayerNorm(x.size()[1:]).to(device)
                x = self.norm2(x).to(device)
        x = self.actv(x)
        x = self.qconv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

    

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        if args.quat_inst_norm:
            self.norm = QuaternionInstanceNorm2d(args.N, num_features,affine=False)
            #self.fc = nn.Linear(style_dim, num_features*2)
        else:
            self.norm = nn.InstanceNorm2d(num_features, affine=False)

        if args.qsngan_layers:
            self.qfc = QuaternionLinear(style_dim, (num_features*2))
        elif args.phm:
            self.qfc = PHMLinear(args.N,style_dim, (num_features*2))
        elif args.htorch_layers:
            self.qfc = layers.QLinear(style_dim//4, (num_features*2)//4)
            #elif args.real:
            #    self.qfc = nn.Linear(style_dim, num_features*2)
        else:
            self.qfc = nn.Linear(style_dim, num_features*2)
        

    def forward(self, x, s):
        #print(s.shape)
        h = self.qfc(s)
        #print(h.shape)
        h = h.view(h.shape[0], h.shape[1], 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)

        if isinstance(x,quaternion.QuaternionTensor):
            return  (1 + gamma) * self.norm(x.torch()) + beta
        return  (1 + gamma) * self.norm(x) + beta



class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        if args.qsngan_layers:
            self.qconv1=QuaternionConv(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
            self.qconv2=QuaternionConv(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        elif args.phm:
            self.qconv1=PHMConv(args.N,dim_in, dim_out, kernel_size=3, stride=1, padding=1)
            self.qconv2=PHMConv(args.N,dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        elif args.htorch_layers:    
            self.qconv1=layers.QConv2d(dim_in//4, dim_out//4, kernel_size=3, stride=1, padding=1)
            self.qconv2=layers.QConv2d(dim_out//4, dim_out//4, kernel_size=3, stride=1, padding=1)
        elif args.real:
            self.qconv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            self.qconv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            if args.qsngan_layers:
                self.qconv1x1 = QuaternionConv(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
            elif args.phm:
                self.qconv1x1 = PHMConv(args.N,dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
           
            elif args.htorch_layers:
                self.qconv1x1=layers.QConv2d(dim_in//4, dim_out//4, kernel_size=1, stride=1, padding=0, bias = False)
            elif args.real:
                self.qconv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)


    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.qconv1x1(x)

        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.qconv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.qconv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

    

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0):
        super().__init__()
        dim_in = 2**14 // img_size
        if args.phm and args.N == 3:
            dim_in = args.img_size
            max_conv_dim = 513
            style_dim = 63
        self.img_size = img_size
        if args.qsngan_layers:
            self.from_rgb = QuaternionConv(4, dim_in, kernel_size=3, stride=1, padding=1)
        elif args.phm and (args.N == 4 or args.N == 2):
            self.from_rgb = PHMConv(args.N,4, dim_in, kernel_size=3, stride=1, padding=1)
        elif args.phm and args.N == 3:
            self.from_rgb = PHMConv(args.N,3, dim_in, kernel_size=3, stride=1, padding=1)
        elif args.htorch_layers:
            self.from_rgb = layers.QConv2d(1, dim_in//4, kernel_size=3, stride=1, padding=1)
        elif args.real:
            self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)

        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        if args.layer_norm:
            self.instance_norm = nn.LayerNorm(dim_in, elementwise_affine=True)
        elif args.quat_inst_norm:
            self.instance_norm = QuaternionInstanceNorm2d(args.N, dim_in, affine=True)
        else:
            self.instance_norm = nn.InstanceNorm2d(dim_in, affine=True)
        self.leaky = nn.LeakyReLU(0.2)
        if args.qsngan_layers:
            self.qconv = QuaternionConv(dim_in, 4, kernel_size=1, stride=1, padding=0)
        elif args.phm and( args.N == 4 or args.N == 2):
            self.qconv = PHMConv(args.N,dim_in, 4, kernel_size=1, stride=1, padding=0)
        elif args.phm and args.N == 3:
            self.qconv = PHMConv(args.N,dim_in, 3, kernel_size=1, stride=1, padding=0)
        elif args.htorch_layers:
            self.qconv = layers.QConv2d(dim_in//4, 1, 1, 1, 0)
        elif args.real:
            self.qconv = nn.Conv2d(dim_in, 3, 1, 1, 0)

        # self.to_rgb = nn.Sequential(
        #     nn.InstanceNorm2d(dim_in, affine=True),
        #     nn.LeakyReLU(0.2),
        #     #nn.Conv2d(dim_in, 4, 1, 1, 0)
        #     layers.QConv2d(dim_in//4, 1, 1, 1, 0)
        #     #layers.QuaternionToReal(dim_in)
        # )
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128,129]):
                cache[x.size(2)] = x
            x = block(x)
        #x = torch.randn(8,512,16,16).to(device)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 129]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear',align_corners=False)
                x = x + self.hpf(mask * cache[x.size(2)])
        
        if (isinstance(x,quaternion.QuaternionTensor)):
            x = x.torch().to(device)
        #x = quaternion.QuaternionTensor(torch.randn(8,128,128,128).to(device))

        #x = torch.randn(8,4,128,128).to(device)

        x = self.instance_norm(x)
        x = self.leaky(x)
        y = self.qconv(x)
        #print(y.torch().is_cuda,y.torch().requires_grad)
        #print(layers.QuaternionToReal(dim_in//4)(y).shape)
        #return quaternion.QuaternionTensor(torch.randn(8,4,128,128).to(device))
        y = y.torch() if isinstance(y,quaternion.QuaternionTensor) else y
        if args.phm and args.N == 3:
            y = F.interpolate(y,args.img_size)
        return y


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers_list = []
        dim_max = 512 
        if args.phm and args.N == 3:
            style_dim = 63
            dim_max=513
            latent_dim = 15
        if args.qsngan_layers:
            layers_list+=[QuaternionLinear(latent_dim, 512)]
        elif args.phm:
            layers_list+=[PHMLinear(args.N,latent_dim, dim_max)]
        elif args.htorch_layers:
            layers_list+=[layers.QLinear(latent_dim//4, 512//4)]
        elif args.real:
            layers_list += [nn.Linear(latent_dim, 512)]

        layers_list+=[nn.ReLU()]
        for _ in range(3):
            if args.qsngan_layers:
                layers_list += [QuaternionLinear(512, 512)]
            elif args.phm:
                layers_list += [PHMLinear(args.N,dim_max, dim_max)]
            elif args.htorch_layers:
                layers_list += [layers.QLinear(512//4, 512//4)]
            elif args.real:
                layers_list += [nn.Linear(512, 512)]

            layers_list += [nn.ReLU()]
        self.shared = nn.Sequential(*layers_list)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            if args.qsngan_layers:
                self.unshared += [nn.Sequential(QuaternionLinear(512, 512),
                                            nn.ReLU(),
                                            QuaternionLinear(512, 512),
                                            nn.ReLU(),
                                            QuaternionLinear(512, 512),
                                            nn.ReLU(),
                                            QuaternionLinear(512, style_dim))]
            elif args.phm:
                self.unshared += [nn.Sequential(PHMLinear(args.N,dim_max, dim_max),
                                            nn.ReLU(),
                                            PHMLinear(args.N,dim_max, dim_max),
                                            nn.ReLU(),
                                            PHMLinear(args.N,dim_max, dim_max),
                                            nn.ReLU(),
                                            PHMLinear(args.N,dim_max, style_dim))]
            elif args.htorch_layers:
                self.unshared += [nn.Sequential(layers.QLinear(512//4, 512//4),
                                            nn.ReLU(),
                                            layers.QLinear(512//4, 512//4),
                                            nn.ReLU(),
                                            layers.QLinear(512//4, 512//4),
                                            nn.ReLU(),
                                            layers.QLinear(512//4, style_dim//4))]
            elif args.real:
                self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h).torch()] if isinstance(h,quaternion.QuaternionTensor) else [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(device)
        s = out[idx, y]  # (batch, style_dim)
        # print(s.shape,"mapping",s.is_cuda)
        # print(out.requires_grad,out.device,idx.requires_grad,idx.device)
        return s

    

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        if args.phm and args.N == 3:
            dim_in = args.img_size
            max_conv_dim = 513
            style_dim=63
        blocks = []
        if args.qsngan_layers:
            blocks += [QuaternionConv(4, dim_in, kernel_size=3, stride=1, padding=1)]
        elif args.phm and (args.N == 4 or args.N == 2):
            blocks += [PHMConv(args.N,4, dim_in, kernel_size=3, stride=1, padding=1)]
        elif args.phm and args.N == 3:
            blocks += [PHMConv(args.N,3, dim_in, kernel_size=3, stride=1, padding=1)]
        
        elif args.htorch_layers:
            blocks += [layers.QConv2d(1, dim_in//4, 3, 1, 1)]
        elif args.real:
            blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]

        if args.qsngan_layers:
            blocks += [QuaternionConv(dim_out, dim_out, kernel_size=4, stride=1, padding=0)]
        elif args.phm:
            blocks += [PHMConv(args.N,dim_out, dim_out, kernel_size=4, stride=1, padding=0)]
        elif args.htorch_layers:
            blocks += [layers.QConv2d(dim_out//4, dim_out//4, 4, 1, 0)]
        elif args.real:
            blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]

        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            if args.qsngan_layers:
                self.unshared += [QuaternionLinear(dim_out, style_dim)]
            elif args.phm:
                self.unshared += [PHMLinear(args.N,dim_out, style_dim)]
            elif args.htorch_layers:
                self.unshared += [layers.QLinear(dim_out//4, style_dim//4)]
            elif args.real:
                self.unshared += [nn.Linear(dim_out, style_dim)]

        
        
    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h).torch()] if isinstance(h,quaternion.QuaternionTensor) else [layer(h)]
        out = torch.stack(out, dim=1) # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        if args.N == 3 and args.phm:
            dim_in = args.img_size
            max_conv_dim = 513

        blocks = []
        self.num_domains = num_domains
        if args.qsngan_layers:
            blocks += [QuaternionConv(4, dim_in, kernel_size=3, stride=1, padding=1)]
        elif args.phm and (args.N == 4 or args.N == 2):
            blocks += [PHMConv(args.N,4, dim_in, kernel_size=3, stride=1, padding=1)]
        elif args.phm and args.N == 3:
            blocks += [PHMConv(args.N,3, dim_in, kernel_size=3, stride=1, padding=1)]
        elif args.htorch_layers:
            blocks += [layers.QConv2d(1, dim_in//4, 3, 1, 1)] 
        elif args.real:
            blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 2
        
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        
        blocks += [nn.LeakyReLU(0.2)]
        if args.qsngan_layers:
            blocks += [QuaternionConv(dim_out, dim_out, kernel_size=4, stride=1, padding=0)]
        elif args.phm:
            blocks += [PHMConv(args.N,dim_out, dim_out, kernel_size=4, stride=1, padding=0)]
        elif args.htorch_layers:
            blocks += [layers.QConv2d(dim_out//4, dim_out//4, 4, 1, 0)]
        elif args.real:
            blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]

        blocks += [nn.LeakyReLU(0.2)]
        if args.phm and args.N == 3:
            while num_domains %3!=0:
                num_domains+=1
        else:
            while num_domains %args.disc_dense!=0:
                num_domains+=1
        if args.qsngan_layers:
            blocks += [QuaternionConv(dim_out, num_domains, kernel_size=1, stride=1, padding=0)]
        elif args.phm:
            blocks += [PHMConv(args.N,dim_out, num_domains, kernel_size=1, stride=1, padding=0)]
        # elif args.phm and args.N == 3:
        #     blocks += [PHMConv(args.N,dim_out, self.num_domains, kernel_size=1, stride=1, padding=0)]
        elif args.htorch_layers:
            blocks += [layers.QConv2d(dim_out//4, num_domains//4, 1, 1, 0)]
        elif args.real:
            blocks += [nn.Conv2d(dim_out, self.num_domains, 1, 1, 0)]
        
        self.main = nn.Sequential(*blocks)
        if not args.real and args.last_dense:
            self.dense = nn.Linear(num_domains,self.num_domains)

    def forward(self, x, y):
        out = self.main(x)
        
        out = out.torch() if isinstance(out,quaternion.QuaternionTensor) else out

        out = out.view(out.size(0), -1)  # (batch, num_domains)
        if not args.real and args.last_dense:
            out = self.dense(out)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        #print(out.shape,"disc",out.is_cuda)
        return out
        # Discriminator real 
        # torch.Size([12, 2, 1, 1]) out
        # torch.Size([12, 2]) out view
        # torch.Size([12]) idx
        # torch.Size([12]) disc

        # Discriminator quaternion domain

        # torch.Size([12, 8, 1, 1]) out
        # torch.Size([12, 8]) out view
        # torch.Size([12]) idx
        # torch.Size([12]) disc


def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    #_exponential moving average
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator.to(device),
                 mapping_network=mapping_network.to(device),
                 style_encoder=style_encoder.to(device),
                 discriminator=discriminator.to(device))
    nets_ema = Munch(generator=generator_ema.to(device),
                     mapping_network=mapping_network_ema.to(device),
                     style_encoder=style_encoder_ema.to(device))

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema