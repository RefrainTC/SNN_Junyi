import numpy as np

import matplotlib.pyplot as plt
from ctypes import *
from numpy.ctypeslib import ndpointer
import time
mylib = CDLL('./test.so')
ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, 
                                      ndim=1,
                                      flags="C")

# define prototypes
mylib.optimize_lr_c.argtypes = [ND_POINTER_1, c_long,c_long,c_long,c_longdouble,c_longdouble]
mylib.optimize_rl_c.argtypes = [ND_POINTER_1, c_long,c_long,c_longdouble,c_longdouble,c_longdouble]
#long optimize_rl_c(double *arr,long idx_start,long idx_end,long double err,long double err_min,long double arr_last)//,long double values[])

#mylib.fun.restype = [c_long]
#c_fun = c_fun.fun #c_sum is the name of our C function
#c_sum.restype = ndpointer(dtype=c_double,shape=(30000000,))

def optimize_even_step(arr):

    l = len(arr)

    idx = l//3

    idx_end = 2*l//3

    last_el = arr[-1]

    sum_left = arr[:idx].sum()

    sum_right = len(arr[idx:])*last_el - arr[idx:].sum()

    while sum_left<sum_right:

        el = arr[idx]

        sum_left+=el

        sum_right-=last_el-el

        idx+=1

    return idx



def optimize_even(arr, idx_list):

    T = len(idx_list)

    #idx_list_full = [0]+idx_list.copy()
    idx_list_full = [0]+idx_list.copy()[:-1]+[len(arr)]
    for idx in range(0,T,2):

        corr = optimize_even_step(arr[idx_list_full[idx]:idx_list_full[idx+2]]-arr[idx_list_full[idx]])

        idx_list_full[idx+1] = idx_list_full[idx]+corr

    return idx_list_full[1:-1]+[len(arr)-1]
    #return idx_list_full[1:]


def intl(arr, T):

   

    l = len(arr)

    idx_step = l//(T+1)

    return [i*idx_step for i in range(1,T+1)]



def intl2(arr, T):

   

    l = len(arr)

    idx_step = l//(2*T+1)

    return [i*idx_step for i in range(1,2*T+1)]






def optimize_lr(arr, idx_list):



    idx = idx_list[0]

    len_r = len(arr[idx:])  

    len_l = len(arr[:idx])

    idx_start = idx-len_l//3

    idx_end = idx+len_r//3

    idx_min = idx_start-1



    len_r = len(arr[idx_min:])

    len_l = len(arr[:idx_min])

    s_left = arr[:idx_min].sum()

    s_right = arr[idx_min:].sum()

    len_lr_diff = len_l-len_r

    s_rl_diff = s_right-s_left



    err_min = arr[idx_start]*(len_l-len_r)+s_right-s_left

    start_time = time.time()
    #print("Inside C code..")# idx_end-idx_start,arr[idx_start],arr[idx_start+1]",idx_end-idx_start,arr[idx_start],arr[idx_start+1])
    index = mylib.optimize_lr_c(arr[idx_start:idx_end+1].astype(np.float64),idx_start-idx_start,idx_end-idx_start,len_lr_diff,s_rl_diff,err_min)
    #print("c index",index+idx_start)
    #print("--- %s seconds after c code---" % (time.time() - start_time))
    start_time = time.time()
    '''
    for idx in range(idx_start, idx_end):

        # len_r-=1

        # len_l+=1

        len_lr_diff +=2

        s_rl_diff -= 2*arr[idx]

        # s_left+=arr[idx]

        # s_right-=arr[idx]

        new_err = arr[idx+1]*len_lr_diff + s_rl_diff



        if new_err < err_min:

            err_min = new_err

            idx_min = idx

   
    print("python index",idx_min)
    print("--- %s seconds after python code---" % (time.time() - start_time))
    return idx_min
    '''
    return index+idx_start


def error_lr(arr,idx_list):

    idx = idx_list[0]

    len_r = len(arr[idx:])

    len_l = len(arr[:idx])

    s_left = arr[:idx].sum()

    s_right = arr[idx:].sum()

    len_lr_diff = len_l-len_r

    s_rl_diff = s_right-s_left



    err_min = arr[idx]*(len_l-len_r)+s_right-s_left

    return err_min

    



def optimize_rl(arr, idx_list):



    idx = idx_list[0]

    len_r = len(arr[idx:])  

    len_l = len(arr[:idx])

    idx_start = idx-len_l//3

    # idx_start = idx

    idx_end = idx+len_r//3

    idx_min = idx_start-1



    len_r = len(arr[idx_min:])

    len_l = len(arr[:idx_min])

    s_left = arr[:idx_min].sum()

    s_right = arr[idx_min:].sum()

    len_lr_diff = len_l-len_r

    s_lr_diff = s_left-s_right

   

    arr_last = arr[-1]

    sum_right = len_r*arr_last

    err_min = s_lr_diff+sum_right

    err = s_lr_diff+sum_right

    start_time = time.time()
    #print("Inside C code..")# idx_end-idx_start,arr[idx_start],arr[idx_start+1]",idx_end-idx_start,arr[idx_start],arr[idx_start+1])
    
    index = mylib.optimize_rl_c(arr[idx_start:idx_end+1].astype(np.float64),idx_start-idx_start,idx_end-idx_start,err,err_min,arr_last)
    #print("c index",index+idx_start)
    #print("--- %s seconds after c code---" % (time.time() - start_time))
    start_time = time.time()

    '''
    for idx in range(idx_start, idx_end):

        # len_r-=1

        # sum_right-=arr_last

        # s_lr_diff += 2*arr[idx]

        err+=2*arr[idx]-arr_last



        if err < err_min:

            err_min = err

            idx_min = idx

   
    print("python index",idx_min)
    print("--- %s seconds after python code---" % (time.time() - start_time))
    return idx_min
    '''
    return index+idx_start


def error_rl(arr, idx_list):

    idx = idx_list[0]

    len_r = len(arr[idx:])

    len_l = len(arr[:idx])

    s_left = arr[:idx].sum()

    s_right = arr[idx:].sum()

    len_lr_diff = len_l-len_r

    s_lr_diff = s_left-s_right

   

    arr_last = arr[-1]

    sum_right = len_r*arr_last

    err_min = s_lr_diff+sum_right

    return err_min



def optimize1(arr, init_idx):

    full_idx = [0]+init_idx.copy()+[len(arr)+1]

    T = len(init_idx)

   

    # for idx in range(T,0,-2):

    #     idx_corr = optimize_lr(arr[full_idx[idx-1]:full_idx[idx+1]]-arr[full_idx[idx-1]],[full_idx[idx]-full_idx[idx-1]])

    #     full_idx[idx] = full_idx[idx-1]+idx_corr



    #     idx_corr = optimize_rl(arr[full_idx[idx-2]:full_idx[idx]]-arr[full_idx[idx-2]],[full_idx[idx-1]-full_idx[idx-2]])

    #     full_idx[idx-1] = full_idx[idx-2]+idx_corr

    # return full_idx[1:-1]

    for idx in range(T,0,-1):

        if idx%2==0:

            idx_corr = optimize_lr(arr[full_idx[idx-1]:full_idx[idx+1]]-arr[full_idx[idx-1]],[full_idx[idx]-full_idx[idx-1]])

            full_idx[idx] = full_idx[idx-1]+idx_corr

        else:

            idx_corr = optimize_rl(arr[full_idx[idx-1]:full_idx[idx+1]]-arr[full_idx[idx-1]],[full_idx[idx]-full_idx[idx-1]])

            full_idx[idx] = full_idx[idx-1]+idx_corr

   

    return full_idx[1:-1]



def optimize(arr, init_idx):

    full_idx = [0]+init_idx.copy() #+[len(arr)+1]

    T = len(init_idx)

   

    # for idx in range(T,0,-2):

    #     idx_corr = optimize_lr(arr[full_idx[idx-1]:full_idx[idx+1]]-arr[full_idx[idx-1]],[full_idx[idx]-full_idx[idx-1]])

    #     full_idx[idx] = full_idx[idx-1]+idx_corr



    #     idx_corr = optimize_rl(arr[full_idx[idx-2]:full_idx[idx]]-arr[full_idx[idx-2]],[full_idx[idx-1]-full_idx[idx-2]])

    #     full_idx[idx-1] = full_idx[idx-2]+idx_corr

    # return full_idx[1:-1]

    for idx in range(T-1,1,-1):

        if (T-1-idx)%2==0:

            if (T-1-idx) == 0:

                idx_corr = optimize_rl(arr[full_idx[idx-1]:],[full_idx[idx]-full_idx[idx-1]])

                full_idx[idx] = full_idx[idx-1]+idx_corr

            else:

                idx_corr = optimize_rl(arr[full_idx[idx-1]:full_idx[idx+1]]-arr[full_idx[idx-1]],[full_idx[idx]-full_idx[idx-1]])

                full_idx[idx] = full_idx[idx-1]+idx_corr

        else:

            idx_corr = optimize_lr(arr[full_idx[idx-1]:full_idx[idx+1]]-arr[full_idx[idx-1]],[full_idx[idx]-full_idx[idx-1]])

            full_idx[idx] = full_idx[idx-1]+idx_corr

    idx_corr = optimize_lr(arr[full_idx[0]:full_idx[2]]-arr[0],[full_idx[1]])

    full_idx[1] = idx_corr

    return full_idx[1:]

def error(arr, idx_list):

    idx_list_full = [0]+idx_list+[len(arr)+1]

    err = 0

    for i in range(0,len(idx_list_full)-2,2):

        err += (arr[idx_list_full[i]:idx_list_full[i+1]]-arr[idx_list_full[i]]).sum()+(idx_list_full[i+2]-idx_list_full[i+1])*arr[idx_list_full[i+2]]-(arr[idx_list_full[i+1]:idx_list_full[i+2]]).sum()

    err += (arr[idx_list_full[-2]:]-arr[idx_list_full[-2]]).sum()

    return err





def thrs_in_out(arr, idx_list):

    thrs_in = []

    thrs_out = []

    # idx_list = [0]+idx_list

    thrs_list = []

    for idx in range(0,len(idx_list)):

        if idx%2==0:

            if idx==0:

                thrs_in.append(arr[idx_list[idx]+1])

            else:

                thrs_in.append(arr[idx_list[idx]+1]-arr[idx_list[idx-1]+1])



        else:

            if idx ==1:

                thrs_out.append(arr[idx_list[idx]+1])

            else:

                thrs_out.append(arr[idx_list[idx]+1]-arr[idx_list[idx-2]+1])

       

    return thrs_in, thrs_out


def test_c():
    c_fun = CDLL('./test.so')

    a = np.abs(np.random.normal(0,1,30000))
    #a = np.zeros((1,30000000))
    a.sort()
    print("gen")
    #
    # print(a)
    indeces = np.arange(0,100)
    T=4
    c_sum = c_fun.fun #c_sum is the name of our C function
    c_sum.restype = ndpointer(dtype=c_double,
                            shape=(30000000,))
    b = c_sum(c_void_p(a.ctypes.data))
    print(b.shape)
