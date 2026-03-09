import pandas as pd
import os
from math import ceil
import numpy as np

class ModelConfig():
    r"""
    This is the configuration class to store the configuration of a [`Model`]. It is used to instantiate an LLM
    model according to the specified arguments, defining the model architecture. 
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
    """
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_ffi = 1,    ## Number of feed forward parallel in the first up projection
        num_encoder_layers=0,
        num_decoder_layers=32,
        num_attention_heads=32,
        head_dim=None,
        num_key_value_heads=None,
        moe_layer_freq = None,
        hidden_act="silu",
        num_experts = 1,
        expert_top_k = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_ffi = num_ffi
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        if head_dim is None:
            head_dim = self.hidden_size // self.num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.moe_layer_freq = moe_layer_freq    ## If n, than every nth value is moe layer.
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        # super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))
      

## Projecting the intermediate weight to smaller (low rank) for compute efficiency -> Tensor network.
def tensorize_ff(dims, kernel_size):
    M, N, K, *others = dims
    if kernel_size > M or kernel_size > N:
        print(f'[Error] Tensorized kernel size [{kernel_size}] to large. Please pick kernel_size < [{min(M, N)}]')
    input = [K, N]
    weight = [M, K]
    output = [M, N]
    input1 = [kernel_size, K//kernel_size, N]
    weight1 = [kernel_size, kernel_size]
    output1 = [kernel_size, K//kernel_size, N]
    weight2 = [K//kernel_size, M//kernel_size]
    output2 = [kernel_size, M//kernel_size, N]
    layer1 = [kernel_size, N * K//kernel_size, kernel_size, *others]
    layer2 = [M//kernel_size, N * kernel_size, K//kernel_size, *others]
    return layer1, layer2

def tensorized_ff1_ff2(layers, kernel_size):
    ff2 = layers.pop()
    ff1 = layers.pop()
    ff1_1, ff1_2 = tensorize_ff(ff1, kernel_size)
    ff2_1, ff2_2 = tensorize_ff(ff2, kernel_size)
    layers.extend([ff1_1, ff1_2, ff2_1, ff2_2])
    return layers
def get_lanugage_model(H, M, N, D, Df):
    key =           [D, N, D, 1, 1, 1, 3]
    value =         [D, N, D, 1, 1, 1, 3]
    query =         [D, M, D, 1, 1, 1, 3]
    logit =         [H, M, N, D//H, 1, 1, 4]
    attend =        [H, M, N, D//H, 1, 1, 5]
    output =        [D, M, D, 1, 1, 1, 3]
    ffo =           [Df, M, D, 1, 1, 1, 3]
    ffi =           [D, N, Df, 1, 1, 1, 3]
    layers = [key, value, query, logit, attend, output, ffo, ffi]
    return layers


def get_lanugage_model_low_rank(H, M, N, D, Df, rank):
    key =           [D, N, D, 1, 1, 1, 3]
    value =         [D, N, D, 1, 1, 1, 3]
    query =         [D, M, D, 1, 1, 1, 3]
    key_proj =      [rank, D, N, 1, 1, 1, 3]
    query_proj =    [rank, D, N, 1, 1, 1, 3]
    logit =         [H, M, rank, D//H, 1, 1, 4]
    attend =        [H, M, rank, D//H, 1, 1, 5]
    output =        [D, M, D, 1, 1, 1, 3]
    ffo =           [Df, M, D, 1, 1, 1, 3]
    ffi =           [D, N, Df, 1, 1, 1, 3]
    layers = [key, value, query, key_proj, query_proj, logit, attend, output, ffo, ffi]
    return layers

def get_lanugage_model_kernel(H, M, N, D, Df, m_ratio):
    key =           [D, N, D, 1, 1, 1, 3]
    value =         [D, N, D, 1, 1, 1, 3]
    query =         [D, M, D, 1, 1, 1, 3]
    key_proj =      [ceil(m_ratio), N, D, 1, 1, 1, 3]
    query_proj =    [ceil(m_ratio), N, D, 1, 1, 1, 3]
    kv =         [H, m_ratio *D//H, N, D//H, 1, 1, 5]
    kqv =        [H, M, m_ratio *D//H, D//H, 1, 1, 5]
    output =        [D, M, D, 1, 1, 1, 3]
    ffo =           [Df, M, D, 1, 1, 1, 3]
    ffi =           [D, N, Df, 1, 1, 1, 3]
    layers = [key, value, query, key_proj, query_proj, kv, kqv, output, ffo, ffi]
    return layers

def get_decoder_model(H, N, D, Df, kv_cache=True, Distributed_cores=1):
    # key =           [D, 1, D, 1, 1, 1, 3]
    # value =         [D, 1, D, 1, 1, 1, 3]
    if kv_cache == True:
        query =         [[3*D//Distributed_cores, 1, D, 1, 1, 1, 3]]
    else:
        query = [[2*D//Distributed_cores,N,D,1,1,1,3],[D//Distributed_cores,1,D,1,1,1,3]]
    if(Distributed_cores <= H):
        logit =         [[H//Distributed_cores, 1, N, D//H, 1, 1, 4]]
        attend =        [[H//Distributed_cores, 1, N, D//H, 1, 1, 5]]
    else:
        logit =         [[1, 1, N, D//H//(Distributed_cores//H), 1, 1, 4],[1, D, 1, 1, 1, 1, 6]]
        attend =        [[1, 1, N, D//H//(Distributed_cores//H), 1, 1, 5],[1, D, 1, 1, 1, 1, 6]]
    assert D%Distributed_cores==0, 'Number of Cores is not such that D can be evenly distributed. Please try again.'
    if Distributed_cores <= H:
        assert H%Distributed_cores==0, 'Number of Cores is not such that #Heads can be evenly distributed. Please try again.'
    else:
        assert (D//(Distributed_cores//H))*H*Distributed_cores==D, 'Number of Cores is not such that #Heads and D can be evenly distributed. Please try again.' 
    output =        [D//Distributed_cores, 1, D, 1, 1, 1, 3]
    all_reduce1 =   [1, D, 1, 1, 1, 1, 6]       ## Scatter and gather
    ffo =           [Df//Distributed_cores, 1, D, 1, 1, 1, 3]
    ffi =           [D, 1, Df//Distributed_cores, 1, 1, 1, 3]
    all_reduce2 =   [1, D, 1, 1, 1, 1, 6]       ## Scatter and gather
    layers = query + logit + attend + [ output, all_reduce1, ffo, ffi, all_reduce2]
    return layers

## TODO:(Instead of doing this; make a class of model config and return it as output.)
def get_configs(name, return_full = False, get_model_config=False):
    # print('Generating config parameters for ', name)
    name = name.lower()
    if name.upper() == 'BERT' or name.lower()== 'vit' or name.lower() == 'swin' or name.lower() == 'gpt-2' or name.lower() == 'opt_125m':      ## ViT,GPT-2 also has same
        D = 768
        H = 12
        Df = 4*D
        model_config = ModelConfig(
            hidden_size=768, num_attention_heads=12, num_ffi = 1,
            intermediate_size=4*768, num_encoder_layers=12
            )
        num_encoder = 12
        num_decoder = 0
    elif name.lower() == 'trxl':
        D = 1024
        H = 16
        Df = 4*D
        num_encoder = 1
        num_decoder = 0
    elif name.lower() == 'xlm':
        D = 2048
        H = 16
        Df = 4*D
        num_encoder = 1
        num_decoder = 0
    elif name.lower() == 'wmt':
        D = 1024
        H = 16
        Df = 4*D
        num_encoder = 12
        num_decoder = 0
    elif name.lower() == 'opt_350m' :
        D = 1024
        H = 16
        Df = 4*D
        num_encoder = 0
        num_decoder = 24
    elif name.lower() == 'gpt-3_1b' or name.lower() == 'opt_1b' :
        D = 2048
        H = 32
        Df = 4*D
        num_encoder = 0
        num_decoder = 24
    elif name.lower() == 'gpt-3_7b' or name.lower() == 'opt_7b' :
        D = 4096
        H = 32
        Df = 4*D
        num_encoder = 0
        num_decoder = 32
    elif name.lower() == 'gpt-3_13b':
        D = 5140
        H = 40
        Df = 4*D
        num_encoder = 0
        num_decoder = 40
    elif name.lower() == 'gpt-3' or name.lower() == 'opt_175b':
        model_config = ModelConfig(
        hidden_size=12288, num_attention_heads=96, num_ffi = 1,
        intermediate_size=4*12288, num_decoder_layers=96,
        )
        D = 12288
        H = 96
        Df = 4*D
        num_encoder = 0
        num_decoder = 96
    elif name.lower() == 'palm':
        model_config = ModelConfig(
            hidden_size=18432, num_attention_heads=48, num_ffi = 1,
            intermediate_size=4*18432, num_decoder_layers=118
            )
        D = 18432
        H = 48
        Df = 4*D
        num_encoder = 0
        num_decoder = 118
    elif name.lower() == 'gemma_7b':
        model_config = ModelConfig(
            hidden_size=3072, num_attention_heads=12, num_ffi = 2,
            intermediate_size=24576, num_decoder_layers=28, head_dim=256
            )
        D = 3072
        H = 16
        Df = 24576
        num_encoder = 0
        num_decoder = 28
    elif name.lower() == 'llama_7b':
        model_config = ModelConfig(
            hidden_size=4096, num_attention_heads=32, num_ffi = 2,
            intermediate_size=11008, num_decoder_layers=32
            )

        D = 4096
        H = 32
        Df = 11008
        num_encoder = 0
        num_decoder = 32
    elif name.lower() == 'llama3_7b':
        model_config = ModelConfig(
            hidden_size=4096, num_attention_heads=32,
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=14336, num_decoder_layers=32,
            )

        D = 4096
        H = 32
        Df = 14336
        num_encoder = 0
        num_decoder = 32
    elif name.lower() == 'llama3_1b':
        model_config = ModelConfig(
            hidden_size=2048, num_attention_heads=32,
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=4*2048, num_decoder_layers=16,
            )

        D = 2048
        H = 32
        Df = 4*2048
        num_encoder = 0
        num_decoder = 16
    elif name.lower() == 'llama_13b':
        model_config = ModelConfig(
            hidden_size=5120, num_attention_heads=40, num_ffi = 2,
            intermediate_size=13824, num_decoder_layers=40
            )

        D = 5120
        H = 40
        Df = 13824
        num_encoder = 0
        num_decoder = 40
    elif name.lower() == 'llama_33b':
        model_config = ModelConfig(
            hidden_size=6656, num_attention_heads=52, num_ffi = 2,
            intermediate_size=17888, num_decoder_layers=60
            )
        D = 6656
        H = 52
        Df = 17888
        num_encoder = 0
        num_decoder = 60
    elif name.lower() == 'opt_30b':
        D = 7168
        H = 56
        Df = 4*D
        num_encoder = 0
        num_decoder = 48
    elif name.lower() == 'llama_70b':
        model_config = ModelConfig(
            hidden_size=8192, num_attention_heads=64, 
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=28672, num_decoder_layers=80,
            )
        D = 8192
        H = 64
        Df = 28672
        num_encoder = 0
        num_decoder = 80
    elif name.lower() == 'mixtral_7x8':
        model_config = ModelConfig(
            hidden_size=4096, num_attention_heads=32, 
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=14336, num_decoder_layers=32,
            expert_top_k=2, num_experts=8, moe_layer_freq=1
            )
    elif name.lower() == 'dbrx':
        model_config = ModelConfig(
            hidden_size=6144, num_attention_heads=48, 
            num_key_value_heads=8, num_ffi = 2,
            intermediate_size=10752, num_decoder_layers=40,
            expert_top_k=4, num_experts=16, moe_layer_freq=1
            )
    elif name.lower() == 'gpt-4':
        x = 84
        model_config = ModelConfig(
            hidden_size=84*128, num_attention_heads=84, 
            num_key_value_heads=84, num_ffi = 1,
            intermediate_size=4*84*128, num_decoder_layers=128,
            expert_top_k=2, num_experts=16, moe_layer_freq=1
            )
    elif name.lower() == 'grok-1':
        model_config = ModelConfig(
            hidden_size=6144, num_attention_heads=48, 
            num_key_value_heads=8, num_ffi = 1,
            intermediate_size=8*6144, num_decoder_layers=64,
            expert_top_k=2, num_experts=8, moe_layer_freq=1
            )
    elif name.lower() == 'super_llm':
        x = 108
        model_config = ModelConfig(
            hidden_size=x*128, num_attention_heads=x, 
            num_key_value_heads=x, num_ffi = 2,
            intermediate_size=4*x*128, num_decoder_layers=128,
            expert_top_k=4, num_experts=32, moe_layer_freq=1
            )
    elif name.lower() == 'trillion_param':
        D = 24*1024
        H = 96
        Df = 96*1024
        num_encoder = 0
        num_decoder = 148
    elif name == 't5s':
        D = 512
        H = 6
        Df = 2048
        num_encoder = 6
        num_decoder = 6
    elif name == 't5b':
        D = 768
        H = 32
        Df = 4*D
        num_encoder = 12
        num_decoder = 12
    elif name == 't5l':
        D = 1024
        H = 32
        Df = 4*D
        num_encoder = 24
        num_decoder = 24
    elif name == 'beit':
        D = 1408
        H = 16
        Df = 6144
        num_encoder = 0
        num_decoder = 40
    elif name == 'llama3_1b':
        # https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
        model_config = ModelConfig(model='meta-llama/Llama-3.2-1B',
    hidden_size=2048, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=4*2048, num_decoder_layers=16,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
    )
    elif name == 'gemma2_2b':
        model_config = ModelConfig(model='google/gemma-2-2B',
    hidden_size=2304, num_attention_heads=8, num_ffi = 2,
    num_key_value_heads=4, head_dim=256,
    intermediate_size=9216, num_decoder_layers=26,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
        )
    elif name == 'llama3_3b':
        model_config = ModelConfig(model='meta-llama/Llama-3.2-3B',
    hidden_size=3072, num_attention_heads=24,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=4*2048, num_decoder_layers=28,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
        )
    elif name == 'llama3_8b':
        model_config = ModelConfig(model='meta-llama/Llama-3.1-8B',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=14336, num_decoder_layers=32,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
        )
    elif name == 'gemma2_27b':
        model_config = ModelConfig(model='google/gemma-2-27B',
    hidden_size=4608, num_attention_heads=32, num_ffi = 2,
    num_key_value_heads=16, head_dim=128,
    intermediate_size=36864, num_decoder_layers=46,
    vocab_size=256000, max_model_len=8*1024, hidden_act="gelu_pytorch_tanh",
    sliding_window=4096,
        )
    elif name == 'qwen2.5_32b':
        model_config = ModelConfig(model='Qwen/Qwen2.5-32B',
    hidden_size=5120, num_attention_heads=40,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=27648, num_decoder_layers=64,
    vocab_size=152064, max_model_len=32*1024, sliding_window=128*1024,hidden_act="silu")

    elif name == 'llama3_70b':
        model_config = ModelConfig(model='meta-llama/Llama-3.1-70B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=28672, num_decoder_layers=80,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
        )
    elif name == 'qwen2.5_72b':
        model_config = ModelConfig(model='Qwen/Qwen2.5-72B',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=29568, num_decoder_layers=80,
    vocab_size=152064, max_model_len=32*1024, sliding_window=128*1024,hidden_act="silu")

    elif name == 'llama3_405b':
        model_config = ModelConfig(model='meta-llama/Llama-3.1-405B',
    hidden_size=16384, num_attention_heads=128,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=3.25*16384, num_decoder_layers=126,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
        )
    else:
        ## If unknown name, then giving parameters of BERT
        print("ERROR, model name parsed incorrect, please check!!! Model Name:",name)
        D = 768
        H = 12
        Df = 4*D
        num_encoder = 1
        num_decoder = 0
    
    if(return_full == True):
        return H,D,Df,num_encoder,num_decoder
    elif get_model_config:
        return model_config
    else:
        return H, D, Df

def create_model(seq_len, name='BERT', data_path='./', method='vanilla', low_rank_ratio=1/8, m_ratio=4, spattn_density=1/16, density=(1.0,1.0,1.0), special_layer_only=False, 
                to_tensorized=False, tensorized_kernel=128, block_size=32,sparisty_budget=0.5,spff_density=1/2,spqkv_density=0.5,
                generate_full_model=False, return_model_dim_only=False):

    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    if(generate_full_model==True):
        H,D,Df, num_enc , num_dec = get_configs(name,return_full=True)
    else:
        # model_config = get_configs(name, get_model_config=True)
        # H = model_config.num_attention_heads
        # D = model_config.hidden_size
        # Df = model_config.intermediate_size
        H, D, Df = get_configs(name)
    M = seq_len
    N = seq_len
    if method == 'vanilla':
        layers = get_lanugage_model(H, M, N, D, Df)
        special_layers = []
    # lin fomer
    elif method == 'lowrank':
        rank = ceil(N*low_rank_ratio)
        layers = get_lanugage_model_low_rank(H, M, N, D, Df, rank)
        special_layers = [3,4, 5, 6]
    #performer model
    elif method == 'kernel':
        layers = get_lanugage_model_kernel(H, M, N, D, Df, m_ratio)
        special_layers = [3, 4, 5, 6]
    # sparse transformer
    elif 'sparse' in method:
        layers = get_lanugage_model(H, M, N, D, Df)
        special_layers = []
        if 'attn' in method:
            special_layers.extend([3, 4])
        if 'FF' in method or 'ff' in method:
            special_layers.extend([6, 7])
        if 'q' in method.lower():
            special_layers.extend([0])
        if 'k' in method.lower():
            special_layers.extend([1])
        if 'v' in method.lower():
            special_layers.extend([2])
        
        if method == 'sparse_op':
            layers = [[D, N, D, 1, 1, 1, 3],[D, N, D, 1, 1, 1, 3]]
            special_layers = [0]
        
        if method == 'sparse':
            special_layers = [3,4]
    elif method == 'pixelfly':
    # We assume sparisty budget and 3rd of that is for the low rank component and rest for butterfly. We choose max k, so that we are still under the budget.
        layers = get_lanugage_model(H, M, N, D, Df)
        special_layers = [3, 4] 
    if to_tensorized:
        layers = tensorized_ff1_ff2(layers, tensorized_kernel)


    # print(special_layers)
    name = name + f'_{method}'

    if special_layer_only:
        layers = np.array(layers)[special_layers]
    
    if(generate_full_model==True):
        layers = np.repeat(layers, num_enc + num_dec, axis=0)
    


    if density:
        densities = np.ones((len(layers), 3), dtype=float) * np.array(density)
        densities[special_layers] = 1.0
        for layer in special_layers:
            
            
            if method == 'sparse_op':
                densities[0][1] = spff_density    # logit output is sparsified
            elif layer < 3:
                densities[layer][1] = spqkv_density
            elif layer == 3:
                densities[3][2] = spattn_density    # logit output is sparsified
            elif layer == 4:
                densities[4][0] = spattn_density    # attend input is sparsified
            elif layer > 4:
                densities[layer][1] = spff_density    # FF weights are sparsified

        if(generate_full_model==True):
            densities = np.repeat(densities, num_enc + num_dec, axis=0)

        df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
        df.to_csv(os.path.join(sparsity_file_path, name + '.csv'),  header=True, index=None)
    
    if(return_model_dim_only==True):
        return layers,densities
    
    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '.csv'),  header=True, index=None)

    return name

def create_inference_prefix_model(input_sequence_length, name='BERT', data_path='./', masked=False,
                         output_gen_tokens=32, **args):
    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity") 
    H, D, Df = get_configs(name)
    N = M = input_sequence_length ## input Seq Len
    Dq = args.get('Dq', D//H)
 


    Hkv = args.get('Hkv', H)
    spff_density = args.get('spff_density',1)
    spwqkv_density = args.get('spwqkv_density',1)
    spwo_density = args.get('spwo_density',1)
    tensor_parallel = args.get('tensor_parallel',1)
    MQA = (Hkv != H)
    
    H = max(ceil(H//tensor_parallel),1)
    Hkv = max(ceil(Hkv//tensor_parallel),1) 
    Df = max(Df//tensor_parallel,1)

    layers = []
    densities = []

    if MQA:
        query =         [[D//tensor_parallel, N, D, 1, 1, 1, 3]]
        keyvalue =      [[2*Hkv*Dq, N, D, 1, 1, 1, 3] ]
    else:
        query =         [[3*D//tensor_parallel, N, D, 1, 1, 1, 3]]
    
    logit =         [[H, M, N, Dq, Hkv, 1, 7 if MQA else 4]]
    attend =        [[H, M, N, Dq, Hkv, 1, 8 if MQA else 5]]

    output =        [[D, M, D//tensor_parallel, 1, 1, 1, 3]]
    ffo =           [[Df, M, D, 1, 1, 1, 3]]
    ffi =           [[D, M, Df, 1, 1, 1, 3]]

    if MQA:
        layers = query + keyvalue + logit + attend  + output + ffo + ffi
    else:
        layers = query + logit + attend + output + ffo + ffi
    
    if 'llama' in name.lower() or 'gemma' in name.lower(): 
        layers += ffo

    # densities.append([1,1,1])
    densities.append([1,spwqkv_density,1])
    densities.append([1,1,1])
    densities.append([1,1,1])
    densities.append([1,spwo_density,1])
    densities.append([1,spff_density,1])
    if 'llama' in name.lower() or 'gemma' in name.lower():
        densities.append([1,spff_density,1])
    densities.append([1,spff_density,1])

    

    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '_prefix'+'.csv'),  header=True, index=None)
    
    df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    df.to_csv(os.path.join(sparsity_file_path, name+ '_prefix' + '.csv'),  header=True, index=None)

    return name+'_prefix'


def create_inference_decode_model(input_sequence_length, name='BERT', data_path='./',
                         output_gen_tokens=32, **args):
    
    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity") 
    H, D, Df = get_configs(name)
    N  = input_sequence_length ## input Seq Len
    Dq = args.get('Dq', D//H)
 
    
    Hkv = args.get('Hkv', H)
    spff_density = args.get('spff_density',1)
    spwqkv_density = args.get('spwqkv_density',1)
    spwo_density = args.get('spwo_density',1)
    tensor_parallel = args.get('tensor_parallel',1)

    MQA = (Hkv != H) 
    H = max(ceil(H//tensor_parallel),1)
    Hkv = max(ceil(Hkv//tensor_parallel),1) 
    Df = max(Df//tensor_parallel,1)

    layers = []
    densities = []

    query =   [[3*D//tensor_parallel, 1, D, 1, 1, 1, 3]]
    
    logit_pre =         [[H, 1, N, Dq, Hkv, 1, 7 if MQA else 9]]
    attend_pre =        [[H, 1, N, Dq, Hkv, 1, 8 if MQA else 10]]
    
    logit_suf =         [[H, 1, output_gen_tokens, Dq, Hkv, 1, 7 if MQA else 4]]
    attend_suf =        [[H, 1, output_gen_tokens, Dq, Hkv, 1, 8 if MQA else 5]]

    output =        [[D, 1, D//tensor_parallel, 1, 1, 1, 3]]
    ffo =           [[Df, 1, D, 1, 1, 1, 3]]    ## Df is already divided
    ffi =           [[D, 1, Df, 1, 1, 1, 3]]
    if 'llama' in name.lower() or 'gemma' in name.lower(): 
        layers = query + logit_pre + logit_suf + attend_pre + attend_suf + output + ffo +ffo + ffi
    else:
        layers = query + logit_pre + logit_suf + attend_pre + attend_suf + output + ffo  + ffi
        
    densities.append([1,spwqkv_density,1])
    densities.append([1,1,1])
    densities.append([1,1,1])
    densities.append([1,1,1])
    densities.append([1,1,1])
    densities.append([1,spwo_density,1])
    densities.append([1,spff_density,1])
    densities.append([1,spff_density,1])
    if 'llama' in name.lower() or 'gemma' in name.lower():
        densities.append([1,spff_density,1])

    

    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '_decode'+'.csv'),  header=True, index=None)
    
    df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    df.to_csv(os.path.join(sparsity_file_path, name+ '_decode' + '.csv'),  header=True, index=None)

    return name+'_decode'

def create_inference_moe_prefix_model(input_sequence_length, name='BERT', data_path='./', masked=False,
                            output_gen_tokens=32,
                            method='vanilla',
                            spattn_density = 0.5,
                            spff_density=0.5,
                            spqkv_density=0.5,
                         **args):
    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity") 
    model_config = get_configs(name, get_model_config=True)
    
    M = N  = input_sequence_length ## input Seq Len

    tensor_parallel = args.get('tensor_parallel',1)

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    ## TODO : Implement the case when moe_layer_freq is >1
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Dq = model_config.head_dim

    MQA = ( Hkv != H)

    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}' 
    H = max(ceil(H/tensor_parallel),1)
    Hkv = max(ceil(Hkv/tensor_parallel),1) 
    Df = max(Df//tensor_parallel,1)

    layers = []
    densities = []


    query =         [[D//tensor_parallel + 2*Hkv*Dq, N, D, 1, 1, 1, 3]]

    logit =         [[H, M, N, Dq, Hkv, 1, 7 if MQA else 4]]
    attend =        [[H, M, N, Dq, Hkv, 1, 8 if MQA else 5]]

    output =        [[D, M, D//tensor_parallel, 1, 1, 1, 3]]

    if moe_layer_freq:
        num_tokens_per_expert = M*K // E
        ffup =           [[E*Df*fi, num_tokens_per_expert, D, 1, 1, 1, 3]]
        ffdown =           [[D, num_tokens_per_expert, E*Df, 1, 1, 1, 3]]
    else:
        ffup =           [[Df*fi, M, D, 1, 1, 1, 3]]
        ffdown =           [[D, M, Df, 1, 1, 1, 3]]


    # if MQA:
    #     layers = query + keyvalue + logit + attend  + output
    # else:
    layers = query + logit + attend + output + ffup + ffdown

    densities = np.ones((len(layers), 3), dtype=float)

    if 'sparse' in method:
        if 'qkv' in method.lower():
            densities[0][1] = spqkv_density
        if 'attn' in method:
            densities[1][0] = spattn_density    # attend input is sparsified
            densities[2][2] = spattn_density    # logit output is sparsified
        if 'o' in method.lower():
            densities[3][1] = spqkv_density
        if 'ff' in method.lower():
            densities[4][1] = spff_density    # FF weights are sparsified
            densities[5][1] = spff_density    # FF weights are sparsified


    df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    df.to_csv(os.path.join(sparsity_file_path, name+ '_prefix' + '.csv'),  header=True, index=None)

    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '_prefix'+'.csv'),  header=True, index=None)

    return name+'_prefix'

def create_inference_moe_decode_model(input_sequence_length, name='BERT', data_path='./',
                         output_gen_tokens=32, **args):
    
    model_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity") 
    
    model_config = get_configs(name, get_model_config=True)
    
    N  = input_sequence_length ## input Seq Len

    tensor_parallel = args.get('tensor_parallel',1)

    D = model_config.hidden_size
    Df = model_config.intermediate_size
    fi = model_config.num_ffi
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    ## TODO : Implement the case when moe_layer_freq is >1
    moe_layer_freq = model_config.moe_layer_freq
    E = model_config.num_experts
    K = model_config.expert_top_k
    Dq = model_config.head_dim

    MQA = ( Hkv != H)
    
    # assert H % tensor_parallel == 0, f'Heads should be equally divisible, H:{H}, TP:{tensor_parallel}' 

    H = max(ceil(H/tensor_parallel),1)
    Hkv = max(ceil(Hkv/tensor_parallel),1) 
    Df = max(Df//tensor_parallel,1)

    layers = []
    densities = []


    query =         [[D//tensor_parallel + 2*Hkv*Dq, 1, D, 1, 1, 1, 3]]

    logit_pre =         [[H, 1, N, Dq, Hkv, 1, 7 if MQA else 9]]
    attend_pre =        [[H, 1, N, Dq, Hkv, 1, 8 if MQA else 10]]
    logit_suf =         [[H, 1, output_gen_tokens, Dq, Hkv, 1, 7 if MQA else 4]]
    attend_suf =        [[H, 1, output_gen_tokens, Dq, Hkv, 1, 8 if MQA else 5]]

    output =        [[D, 1, D//tensor_parallel, 1, 1, 1, 3]]
    ffup =           [[K*Df*fi, 1, D, 1, 1, 1, 3]]    ## Df is already divided
    ffdown =           [[D, 1, K*Df, 1, 1, 1, 3]]

    ffup_unused =   [[(E-K)*Df*fi, 0, D, 1, 1, 1, 3]]   
    ffdown_unused =   [[D, 0, (E-K)*Df, 1, 1, 1, 3]] 

    layers = query + logit_pre + logit_suf + attend_pre + attend_suf + output
    
    
    # for _ in range(fi):
    layers += (ffup + ffup_unused) if moe_layer_freq  else ffup
    layers += (ffdown + ffdown_unused)  if moe_layer_freq  else ffdown
        
    densities = np.ones((len(layers), 3), dtype=float) 
    
    df = pd.DataFrame(densities,columns=['I', 'W', 'O'])
    df.to_csv(os.path.join(sparsity_file_path, name+ '_decode' + '.csv'),  header=True, index=None)

    

    df = pd.DataFrame(layers, columns=['M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    df.to_csv(os.path.join(model_path, name + '_decode'+'.csv'),  header=True, index=None)
    
    return name+'_decode'

if __name__ == '__main__':
    model = 'BERT'
    model_path = os.path.join('../',"data/model/language")
    create_model(256, name=model, model_path=model_path)