#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""
#!/usr/bin/env python
"""

import argparse
#######################################################
# parse CLI arguments 
# set CLI in launch.json for vscode
# set CLI with -xxx for colab
######################################################

def parse_arg(): # parameter not used yet
    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-p", '--process', type=str, choices=['load', 'transfert', 'maker'], help='optional either load and run model or generate one with transfert learning or with model maker. default load', required=False, default="load")
    parser.add_argument("-a", '--app', type=str, help='optional. application. default lili.', required=False, default='lili')
    parser.add_argument("-m", '--model', type=str, choices=['tf', 'li', 'h5'], help='optional. tensorflow model used for run. default tf.', required=False, default='tf')
    parser.add_argument("-q", '--quant', type=str, choices=['fp32', 'default', 'adapt', 'quant-fp', 'TPU', 'GPU', '18by8', 'TPU_edgetpu'], help='optional. tensorflow lite quantization used for run. default fp32.', required=False, default = 'fp32')
    parser.add_argument("-t", '--threshold', type=int, help='optional. threshold in %. default 60%.', required=False, default = 60)

    # The store_true option automatically creates a default value of False.   
    parser.add_argument("-e", '--headless', help='optional. run in a headless, ie no screen. default FALSE (ie screen)', required=False, action="store_true")

    # returned argument is boolean. action="store_true" means default is False
    # nothing, 'windowed':False
    # -w  'windowed': True
    # {'process': 'load', 'app': 'lili', 'model': 'li', 'quant': 'GPU', 'threshold': 60, 'windowed': False, 'benchmark': True}
    

    parser.add_argument("-b", '--benchmark', help='optional. default False (ie do not run benchmarks)', required=False, action="store_true")

    # return from parsing is NOT a dict, a namespace . can access as parsed_args.new if stay as namespace
    parsed_args=parser.parse_args() 
    
    parsed_args = vars(parsed_args) # convert object to dict. get __dict__ attribute
    print('ARG: parsed argument is a  dictionary: ', parsed_args)
    
    """
    print('ARG: keys:' , end = ' ')
    for i in parsed_args.keys():
        print (i , end = ' ')
    print('\n')
    """
    
    return(parsed_args) # dict

