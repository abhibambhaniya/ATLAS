import os, sys
script_dir = os.getcwd()
print(script_dir)

module_path = script_dir

for _ in range(1):
    module_path = os.path.abspath(os.path.join(module_path, '../'))
    # print(module_path)
    if module_path not in sys.path:
        sys.path.insert(0,module_path)
    if os.path.basename(module_path) =='ATLAS':
        break

print(module_path)

from src.unit import Unit
from src.operators import *
import src.operators
from src.analye_model import get_model_df
from src.analye_model import *
from src.system import System
import matplotlib.pyplot as plt
from utils.plot_rooflines import *

# Compute power of two greater than or equal to `n`
def findNextPowerOf2(n):

    if(n == 1):
        return n
    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1

    # do till only one bit is left
    while (n & n - 1) and n > 1:
        n = n & n - 1       # unset rightmost bit

    # `n` is now a power of two (less than `n`)

    # return next power
    if n > 1:
        return n << 1
    else:
        return 1
    # else:
    #     return 0

def get_memory_inst_area(num_banks= 1024, Rd_BW_required=1024):
    ## All banks are of size 32kB.
    if(Rd_BW_required <= num_banks*16):         ## BW = 16B per bank
        return 16,0.026196
    elif(Rd_BW_required <= num_banks*32):
        return 32,0.0275925
    elif(Rd_BW_required <= num_banks*64):
        return 64,0.031689
    else:
        print("Very high BW required, generating a random guessed number.")
        return 128,0.040

def get_mem_wrapper_crossbar_area(num_mem_inst_per_wrappers=256,memory_instance_BW=16):
    scaling_ratio = 60
    ## These area are in m2 so changing to mm^2
    mem_wrapper_crossbar_16B_area = {2:0.000000000396493,4:0.000000000792986,8:0.000000001585972,16:0.00000000317194,32:0.00000000634406,64:0.00000001268837,128:0.00000002537687,256:0.0000000542376,512:0.0002276458433,1024:5e-7}
    mem_wrapper_crossbar_32B_area = {2:0.000000001585976,4:0.00000000317194,8:0.00000000634388,16:0.00000001268786,32:0.00000002537603,64:0.00000005075235,128:0.0000001074763,256:0.0000002149516,512:1e-6,1024:2.5e-6}
    mem_wrapper_crossbar_64B_area = {2:0.000000006343884,4:0.0000000126878,8:0.00000002537563,16:0.00000005075143,32:0.0000001015032,64:0.0000002129584,128:0.0000004318867,256:0.000000853817,512:2e-6,1024:5e-6 }
    # print("For mem wrapper CB area",num_mem_inst_per_wrappers,memory_instance_BW)
    if(num_mem_inst_per_wrappers == 1.0 or num_mem_inst_per_wrappers == 0):
        return 0
    if(memory_instance_BW==16):
        return mem_wrapper_crossbar_16B_area[num_mem_inst_per_wrappers]*1e6*scaling_ratio
    elif(memory_instance_BW==32):
        return mem_wrapper_crossbar_32B_area[num_mem_inst_per_wrappers]*1e6*scaling_ratio
    elif(memory_instance_BW==64):
        return mem_wrapper_crossbar_64B_area[num_mem_inst_per_wrappers]*1e6*scaling_ratio
    else:
        print("wrapper crossbar_area_not_found,returning random value")
        return 1e-5
def get_mem_to_compute_crossbar_cost(num_wrappers=64,memory_instance_BW=16):
    scaling_ratio = 60
    ## These area are in m2 so changing to mm^2
    ## DSENT numbers
    mem_to_compute_crossbar_16B_area = {2:0.000000000792986,4:0.000000003171942,8:0.00000001268777,16:0.00000005075106,32:0.0000002030102,64:0.0000008120554,128:0.000003248239,256:0.00001388482,512:0.00005690014953,1024:0.0002304974882,2048:4.5*0.0002304974882}
    mem_to_compute_crossbar_32B_area = {2:0.000000003171942,4:0.00000001268776,8:0.00000005075104,16:0.00000020300616,32:0.00000081203336,64:0.000003248148,128:0.0000137569584,256:0.00005625547096,512:0.0002276458433,1024:0.0009159372297}
    mem_to_compute_crossbar_64B_area = {2:0.00000001268773,4:0.00000005075112,8:0.0000002030052,16:0.0000008120223,32:0.000003248103,64:0.00001362941,128:0.00005528149,256:0.0002185766,512:0.0008713326227,1024:0.00347859853}
    
    if num_wrappers == 3:
        num_wrappers = 4
    # print("For mem to compute area:",num_wrappers,memory_instance_BW)
    if num_wrappers == 1 or num_wrappers == 0:
        return 0
    elif num_wrappers > 2 and num_wrappers < 1024 and (num_wrappers & (num_wrappers - 1)) != 0:
        x = 2
        while x < num_wrappers:
            y = num_wrappers - x
            if (y & (y - 1)) == 0:  # Check if y is a power of 2
                break
            x *= 2
        if x % 2 == 0 and y % 2 == 0:
            return get_mem_to_compute_crossbar_cost(x,memory_instance_BW) + get_mem_to_compute_crossbar_cost(y,memory_instance_BW)
        else:
            print("mem to compute crossbar_area_not_found,returning random value")
            return mem_to_compute_crossbar_64B_area[num_wrappers]*1e6*(2**(memory_instance_BW/64))
    elif(memory_instance_BW==16):
        return mem_to_compute_crossbar_16B_area[num_wrappers]*1e6*scaling_ratio
    elif(memory_instance_BW==32):
        return mem_to_compute_crossbar_32B_area[num_wrappers]*1e6*scaling_ratio
    elif(memory_instance_BW==64):
        return mem_to_compute_crossbar_64B_area[num_wrappers]*1e6*scaling_ratio
    else:
        print("mem to compute crossbar_area_not_found,returning random value")
        return mem_to_compute_crossbar_64B_area[num_wrappers]*1e6*(2**(memory_instance_BW/64))
def get_structured_compute_cost(M_ratio=1, num_PEs=16384):
    # We are using post route numbers for structured and unstructured hardwares. 
    # Link to the doc with all the numbers : https://docs.google.com/spreadsheets/d/1TgsUtCn4XoOks7cGxgD1CZE5ZNwijX8ZZcTvRXidJ1k/edit?usp=sharing
    # print(M_ratio,num_PEs)
    compute_cost = (0.5946497*M_ratio + 23.46993)*(num_PEs/16384)*0.5
    return compute_cost

def get_unstructured_compute_cost(num_PEs=16384):
    # For unstrucutred HW we use SIGMA as our base, and calculate sparsity using the number of PEs in a Single PU of SIGMA.
    # We keep default as 128 since we that gives a good optima between area and energy and performance trade-off according to the paper
    # Paper link: https://cpn-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/01/sigma_hpca2020.pdf 
    # Refer Fig 9, for DSE of various design points.

    # We are using post route numbers for structured and unstructured hardwares. 
    # Link to the doc with all the numbers : https://docs.google.com/spreadsheets/d/1TgsUtCn4XoOks7cGxgD1CZE5ZNwijX8ZZcTvRXidJ1k/edit?usp=sharing
    cost = (44.29944384/128)*(num_PEs/128)*0.6
    return cost

def get_distributed_noc_cost(num_cores=256,link_BW=16):

    if(num_cores == 1):
        return 0
    elif(np.emath.logn(2,num_cores)%2 == 0):
        num_rows = int(np.sqrt(num_cores))
        num_cols = num_rows
    else:
        num_rows = int(np.sqrt(num_cores/2))
        num_cols = num_rows * 2

    loop_16B_area = { 1:0,  2:0, 4:0, 8:0, 16:0, 32:0 }
    ## Area for Horizontal loops
    area  = loop_16B_area[num_cols]*num_rows
    ## Area for vertical loops
    area += loop_16B_area[num_rows]*num_cols
    ## Area scales linearly with link Bandwidth
    area *= (link_BW/16)
    return area


def get_HW_cost(system,sparse_pe_support=0.5,print_distribution=False,unstructure_BW=4):
    # For strucutred Hardware we assume N:M sparsity, Where M should be a power of 2.
    # For a given M, N can be any power of 2 smaller than and equal to M.
    # Eg: For density = 0.0625(Sparsity = 0.9375) , M will be 16, then the HW can support 1:16,2:16,4:16,8:16 and 16:16. 
    # M_ratio = 1/sparse_pe_support
    if(system.accelerator_type=="unstructured"):
        M_ratio = unstructure_BW                 # For unstructured we assume fixed BW that is 4x/8x that of dense and we get (75/80%) pe utilization for random case and (60/65%) for worst case
    elif sparse_pe_support == 1 :
        M_ratio = 1
    # elif(sparse_pe_support<0.015625):
    #     return 
    else:
        M_ratio = findNextPowerOf2(int(1/sparse_pe_support)) 
    ## assuming a square systolic-like array
    # print("sparsity:",sparse_pe_support,", M_ratio:",M_ratio)

    if(system.mxu_shape is None):           ## Assume Monotlithic system
        num_pes = findNextPowerOf2(int(system.op_per_sec/system.frequency))
        Num_cores = 1
        Num_PEs_per_dim = ceil(np.sqrt(num_pes))
        Non_sparse_PE_dim = Num_PEs_per_dim 
    else:
        ## Unequal Dim
        Num_PEs_per_dim  = system.mxu_shape[-2]
        Non_sparse_PE_dim = system.mxu_shape[-1] 
        num_pes = np.prod(system.mxu_shape[-2:])
        ## Mem Per Core
        Num_cores = np.prod(system.mxu_shape[:-2])
        

    
    Total_mem = (system.on_chip_mem_size if (system.on_chip_mem_size != float('Inf')) else 32e6) / Num_cores      ## default assumed 32MB


    
    
    
    Rd_BW_required = Num_PEs_per_dim*(M_ratio)*system.get_bit_multiplier('M')
    Non_sparse_BW_required = Non_sparse_PE_dim*system.get_bit_multiplier('M') 
    # BW_per_mem_wrapper = Rd_BW_required *1e6 / Total_mem            ## This is in terms of B/cycle per MB
    # print(Rd_BW_required)

    
    ## Default multification factor is 10^x, so changing to 2^x for getting number of instances.
    num_mem_insts = (Total_mem*1024*1024/1e6) / (32*1024)                    ## We assume smallest bank of 32kB and increase the width size as required.


    memory_instance_BW,memory_instance_area = get_memory_inst_area(num_mem_insts,Rd_BW_required)
    
    num_wrappers = findNextPowerOf2(int(Rd_BW_required //memory_instance_BW ))
    num_mem_inst_per_wrappers = findNextPowerOf2(int(num_mem_insts/num_wrappers))
    
    # 

    mem_wrapper_crossbar_area = get_mem_wrapper_crossbar_area(num_mem_inst_per_wrappers,memory_instance_BW) 
    # print("Wrapper CB:", mem_wrapper_crossbar_area)

    if(system.accelerator_type=="unstructured"):
        compute_cost = get_unstructured_compute_cost(num_pes)
    else:
        compute_cost = get_structured_compute_cost(M_ratio,num_pes)

    mem_inst_cost = memory_instance_area*num_mem_insts
    mem_wrapper_mux_cost = mem_wrapper_crossbar_area * num_wrappers
    mem_compute_crossbar_cost = get_mem_to_compute_crossbar_cost(num_wrappers,memory_instance_BW) + get_mem_to_compute_crossbar_cost(int(num_wrappers*Non_sparse_BW_required/Rd_BW_required), memory_instance_BW)

    # print("Compute CB:", mem_compute_crossbar_cost)
    ## This Cost of 1 core
    cost = compute_cost + mem_inst_cost + mem_wrapper_mux_cost + mem_compute_crossbar_cost

    ## Now for connection between cores
    ICN_cores_area = get_distributed_noc_cost(Num_cores, 16) 

    final_cost = Num_cores * cost + ICN_cores_area
    if(print_distribution):
        print(f"For Num_Pes:{num_pes}, Num_cores:{Num_cores}")
        print("M ratio:",M_ratio," , num_inst:",num_mem_insts,", num wrappers:",num_wrappers,", num_inst per wrapper:",num_mem_inst_per_wrappers)
        print("Rd_BW_required:" ,Rd_BW_required,", memory_instance_BW:",memory_instance_BW, ", num mem insts:",num_mem_insts, ", Num wrappers:",num_wrappers, ", num_inst per wrapper:",num_mem_inst_per_wrappers)
        print(compute_cost , mem_inst_cost , mem_wrapper_mux_cost , mem_compute_crossbar_cost)
    
        print(f" C:{Num_cores*100*compute_cost/final_cost:0.02f} , M:{Num_cores*100*mem_inst_cost/final_cost:0.02f} , ICN:{Num_cores*100*(mem_wrapper_mux_cost+mem_compute_crossbar_cost)/final_cost:0.02f}")
    return final_cost,M_ratio

def merit_function( speedup, area, energy):
    ## Note, Area and Energy should be normalized.
    performance = speedup
    TCO = area +   4 * energy 
    return performance/TCO

def plot_cost_speedup_sweep(unit,system,model='WMT',seq_len=256,batch_size = 1, FLAT=True, minimum_attn_density = 0.1, minimum_ff_density=0.5 ,method='sparse_attn',plot_graph=True,print_value=False,debug=False,latency_threshold=1,use_quality=False):
    BWs= []
    BW_speedup = []
    cost = []
    total_latency = []
    sparse_pe_support_array = []
    
    if(method == 'sparse_attn'):
        minimum_ff_density = 1
    elif(method == 'sparse_ff'):
        minimum_attn_density = 1
    elif(method != 'sparse_attn' and method != 'sparse_ff' and method != 'sparse_attn_ff'):
        print('Invalid sparsity method!! Acceptable methods are sparse_attn,sparse_ff , sparse_attn_ff')
    minimum_model_density = min(minimum_attn_density,minimum_ff_density) 
    ## This array has density in the model that is supported by the PE.
    ## The first value of this array is considered as baseline for speedup
    if(system.accelerator_type == "structured"):
        # pe_sparsity_sweep = [1,0.5,0.25,0.125,0.0625,0.03125,0.015625,1/128,0.0001]
        pe_sparsity_sweep = [1,0.5,0.25,0.125,0.0001]
    else:
        pe_sparsity_sweep = [1,0.00001]
    system_is_structured = (system.accelerator_type == "structured") 
    if(print_value):
        print("Seq Len = ", str(seq_len), ", Flat = ",str(FLAT) , " , minimum_model_density = ", minimum_model_density)
    for sparse_pe_support in pe_sparsity_sweep:
        if(sparse_pe_support==0.0001 and system_is_structured):
            system.accelerator_type = "unstructured"
        #####Performance modelling#####
        data_path = os.path.join(module_path,"data")
        # print(seq_len,model,data_path,method,minimum_model_density)
        generated_model = create_model(seq_len=seq_len, name=model, data_path=data_path,method=method,spattn_density=minimum_attn_density,spff_density=minimum_ff_density)
        batch_size = batch_size

        system.set_pe_min_density_support(sparse_pe_support)
        # system.set_onchip_mem_bw(unit,(1000*(1+1/sparse_pe_support)))
        model_df = get_model_df(generated_model, system, unit, batch_size, data_path, sparse=True,intermediate_on_chip=FLAT )
        if(debug):
            dot_roofline(model_df,system=system,unit=unit)                                                                          ## Uncomment for debug
            display(model_df)                                                                                             ## Uncomment for debug 
            print("Num of cycles(M) for the layer :",get_summary_table(model_df,system,unit).loc[0,'Cycles']/1000000)                             ## Uncomment for debug 
    
        ## This gives out number of cycles for the execution of the whole layer
        latency_for_current_HW=get_summary_table(model_df,system,unit).loc[0,'Cycles']/1000
        total_latency.append(latency_for_current_HW) 
        
        #speedup = (100*(total_latency[0] - latency_for_current_HW)/total_latency[0] )
        speedup = (total_latency[0]/latency_for_current_HW)
        BW_speedup.append(speedup)

        ##### Cost analysis######
        sparse_pe_support_array.append(sparse_pe_support)
        onchip_BW_cost,M_ratio = get_HW_cost(system,sparse_pe_support)  

         
        BWs.append(onchip_BW_cost)
        onchip_BW_cost =  (onchip_BW_cost)/BWs[0]

        #### Quality Cost ######    
        # 2:2	31.040497
        # 1:2	30.762443
        # 1:4	29.860972
        # 1:8	28.892418
        # 1:16	28.331074
        # 1%	
        # 2%	31.141843
        # 5%	31.3224
        # 12.5	31.405631
        # 25%	31.358203
        if(use_quality==True):
            if((system.accelerator_type == "structured") and (sparse_pe_support < 0.25)):
                speedup=0
            elif((system.accelerator_type == "unstructured") and (minimum_ff_density>0.02)):
                speedup=speedup*1.2

        if(print_value):
            # print("Density supported = ",sparse_pe_support, " , Speedup wrt dense hardware = ",speedup, " , Speed/cost:" , speedup/onchip_BW_cost)
            print(M_ratio, total_latency[-1],onchip_BW_cost, speedup/onchip_BW_cost)
        cost.append(speedup/onchip_BW_cost)
         
        if(system_is_structured):
           system.accelerator_type = "structured" 

    if(plot_graph):
        fig, ax1 = plt.subplots() 
        # plot_1 = ax1.plot(sparse_pe_support_array,BW_speedup,c='green',label='Abs. Speedups')
        plot_1 = ax1.plot(sparse_pe_support_array,total_latency,c='green',label='Runtime')
        plot_2 = ax1.plot(sparse_pe_support_array,cost,label='Merit')
        # ax1.yscale("log")
        ax1.set_ylabel("Speedup*Cost",c='green')
        ax1.tick_params(axis ='y', labelcolor = 'green') 
        ax1.set_xlabel("PE sparsity")


    max_cost = cost[0]      ## this indicates the merit of dense
    for i in range(len(cost)):
        if ((cost[i] >= max_cost) and (total_latency[i] <= latency_threshold*total_latency[0])):
            max_cost = cost[i]
            
    for i in range(len(cost)):
        if ((cost[i] == max_cost) and (total_latency[i] <= latency_threshold*total_latency[0])):
            # print (i, cost[i] )
            if(plot_graph):
                ax1.scatter(sparse_pe_support_array[i] ,max(cost))
                ax1.text(sparse_pe_support_array[i], max(cost)+0.1,str(sparse_pe_support_array[i]), size=16)
            break

    if(total_latency[i] > latency_threshold*total_latency[0]):
        return "Not Possible"

    if(plot_graph):
        ax2 = ax1.twinx() 
        plot_3 = ax2.plot(sparse_pe_support_array,BWs,c='red', label='Area')
        ax2.set_ylabel("Area cost",c='red')
        ax2.tick_params(axis ='y', labelcolor = 'red') 
        # Add legends

        lns = plot_1 + plot_2 + plot_3
        lns =  plot_2 
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc='upper right')

        plt.title('WITH FLAT='+ str(FLAT) +' , Seq Len = ' + str(seq_len)+', minimum_model_density = '+ str(minimum_model_density) + ' Optimal sparsity needed = '+str(sparse_pe_support_array[i]))
    if(sparse_pe_support_array[i]==1):
        return "Dense"
    elif(sparse_pe_support_array[i]==0.5):
        return "1:2"
    elif(sparse_pe_support_array[i]==0.25):
        return "1:4"
    elif(sparse_pe_support_array[i]==0.125):
        return "1:8"
    elif(sparse_pe_support_array[i]==0.0625):
        return "1:16"
    elif(sparse_pe_support_array[i]==0.0001):
        return "Unstructured"
    return sparse_pe_support_array[i]
