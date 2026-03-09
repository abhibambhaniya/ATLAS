import os, sys
script_dir = os.path.dirname(__file__)
module_path = script_dir
for _ in range(1):
    if os.path.basename(module_path) =='ATLAS':
        break
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    if module_path not in sys.path:
        sys.path.insert(0,module_path)

from src.unit import Unit
from src.operators import *
import src.operators as operators
from src.operator_base import op_type_dicts
from src.system import System
import pandas as pd
import os
from utils.get_language_model import *
# from src.sparse_hw_cost import *

def get_attn_index(df):
    ret = []
    for idx in range(len(df)):
        if 'Attend' in df.loc[idx, 'Op Type'] or 'Logit' in df.loc[idx, 'Op Type']:
           ret.append(idx)
    return ret

def get_summary_table(df,system,unit,sparse_pe_support=1, model_characterstics=False):
    if model_characterstics == False:
        total_cycles = np.sum(df['Cycles'])
        total_latencies = np.sum(df['Latency (msec)'])
        total_energy = np.sum(df['Total energy (mJ)'])

    attn_idx = get_attn_index(df)
    total_parameters = np.sum(df['Input_w (MB)']) - sum([df.loc[i, 'Input_w (MB)'] for i in attn_idx])
    total_data = np.sum(df['Input_a (MB)'] + df['Input_w (MB)'] + df['Output (MB)']) 
    total_MACS = np.sum(df['Num ops (MFLOP)'])
    # total_weights = np.sum(df['Input_w (MB)'])
    total_weights = 0; 
    for i in range(len(df)):
        if ('Logit' not in df.loc[i, 'Op Type']  and 'Attend' not in df.loc[i, 'Op Type']):
           total_weights = total_weights + df.loc[i,'Input_w (MB)'] 
    max_memory_footprint = max([df.loc[i, 'Input_a (MB)'] + df.loc[i, 'Input_w (MB)'] + df.loc[i, 'Output (MB)'] for i in range(len(df))])
    
    # Total_mem = df.loc[0,'Input_a (MB)'] + df.loc[len(df)-1,'Output (MB)'] +total_weights 
    # Mem_time = unit.unit_to_raw(Total_mem * system.get_bit_multiplier(type='M'), type='M') /system.offchip_mem_bw
    # Compute_time = unit.unit_to_raw(total_MACS, type='O')/system.flops 

    # print("Assuming All as a single model and thus have only input to the first layer and output of the last layer")
    # print("Memory Time(ms):",unit.raw_to_unit(Mem_time,type='T'))
    # print("Compute Time(ms)",unit.raw_to_unit(Compute_time,type='T'))
    # print("Real execultion time", total_latencies)
    ret = {
            f'MACs ({unit.unit_flop})': [total_MACS],
            f'Total Data ({unit.unit_mem})': [total_data],
            f'Total Weights ({unit.unit_mem})': [total_weights],
            f'Parameters  ({unit.unit_mem})': [total_parameters],
            f'On-chip Memory Footprint ({unit.unit_mem})': [max_memory_footprint],
        }
    if model_characterstics == False:
        ret.update({
            f'Latency ({unit.unit_time})': [total_latencies],
            'Cycles': [total_cycles],
            'Energy': [total_energy],
        })
    
        
    return pd.DataFrame.from_dict(ret)

def analysis_model(model_dims, system=None, unit=Unit(), densities = None,intermediate_on_chip=False,
                   beam_size=1, beam_merge=False, model_characterstics=False):
    roofline_list = []
    if densities is None:
       densities = np.ones((len(model_dims), 3), dtype=float) 
    for i, (dim, density) in enumerate(zip(model_dims, densities)):
        type = op_type_dicts[dim[-1]]
        operator = getattr(operators, type)
        if beam_merge:
            if dim[-1] == 9 or dim[-1] == 10:
                dim[0] /= beam_size
        operator_instance = operator(dim=dim, density=density)
        # print(density[0],density[1],density[2])
        if (intermediate_on_chip):
            if(type == 'Logit' or type == 'Logit_MQA'):
                operator_instance.set_mem_pin(output='on')    
            elif(type == 'Attend'or type == 'Attend_MQA'):
                operator_instance.set_mem_pin(input_a='on')
            

        if model_characterstics:
            roofline = operator_instance.get_model_characterstics(system=system, unit=unit) 
        else:
            roofline = operator_instance.get_roofline(system=system, unit=unit)

        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    # pd.set_option("precision", 3)
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)

    # df.style.format('{:.2f}')
    # df.to_csv('output/trial.csv')
    return df


def get_model_df(model, system, unit, batch_size=1, data_path='./', sparse=True, intermediate_on_chip=False,
                beam_size=1, beam_merge=False, model_characterstics=False):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model + ".csv")
    density_file = os.path.join(sparsity_file_path, model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size                    # Batch size has been fixed to 1.
    model_defs = np.append(batch_sizes, model_defs, axis=1).astype(int)

    densities = np.ones((len(model_defs), 3), dtype=float)
    if sparse:
        try:
            df = pd.read_csv(density_file)
            density_defs = df.to_numpy()
            densities[:len(density_defs),:] = density_defs
        except:
            print('[INFO]Use default dense analysis.')



    model_df  = analysis_model(model_defs, system, unit, densities, intermediate_on_chip, beam_size, beam_merge, model_characterstics)
    return model_df

def get_optimal_bw_df(model, system, unit, batch_size=1, data_path='./', sparse=True, intermediate_on_chip=False):
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model + ".csv")
    density_file = os.path.join(sparsity_file_path, model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    batch_sizes = np.ones((len(model_defs), 1)) * batch_size                    # Batch size has been fixed to 1.
    model_defs = np.append(batch_sizes, model_defs, axis=1).astype(int)

    densities = np.ones((len(model_defs), 3), dtype=float)
    if sparse:
        try:
            df = pd.read_csv(density_file)
            density_defs = df.to_numpy()
            densities[:len(density_defs),:] = density_defs
        except:
            print('[INFO]Use default dense analysis.')

    
    
    roofline_list = []
    for i, (dim, density) in enumerate(zip(model_defs, densities)):
        type = op_type_dicts[dim[-1]]
        operator = getattr(operators, type)
        operator_instance = operator(dim=dim, density=density)
        # print(density[0],density[1],density[2])
        if (intermediate_on_chip):
            if(type == 'Logit'):
                operator_instance.set_mem_pin(output='on')
            elif(type == 'Attend'):
                operator_instance.set_mem_pin(input_a='on')
        roofline = operator_instance.get_roofline(system=system, unit=unit)
        # print("i = ", i)
        while(roofline['C/M ratio'] > 1.05 or roofline['C/M ratio'] < 0.95 ):
            system.set_offchip_mem_bw(unit,system.get_offchip_mem_bw(unit)/roofline['C/M ratio'])
            roofline = operator_instance.get_roofline(system=system, unit=unit) 
        # print(roofline['C/M ratio'],",BW =",system.get_offchip_mem_bw(unit))
        roofline['BW'] = system.get_offchip_mem_bw(unit) 
        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    # pd.set_option("precision", 3)
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    model_df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)


    # model_df  = analysis_model(model_defs, system, unit, densities,intermediate_on_chip)
    return model_df


if __name__ == '__main__':

    # model = 'example'
    # data_path = os.path.join(module_path,"data/")
    #

    method = 'sparse'
    low_rank_ratio = 1/8
    m_ratio = 4
    spattn_density = 0.1
    seq_len = 256
    batch_size = 4
    model = 'BERT'
    unit = Unit()
    system = System(unit, mxu_shape = [4, 128, 128], compress_mem=True, skip_compute=True, skip_compute_on_noopt_output=True)
    data_path = os.path.join(module_path,"data")
    model_path = os.path.join(data_path,"model")
    create_model(seq_len, name=model, data_path=data_path, density=(1,1,1), low_rank_ratio=low_rank_ratio,
                 m_ratio=m_ratio, spattn_density=spattn_density, method=method, special_layer_only=False,
                 to_tensorized=True)
    model_name = model + f'_{method}'

    model_df = get_model_df(model_name, system, unit, batch_size, data_path)
    get_summary_table(model_df)
    print(model_df)

