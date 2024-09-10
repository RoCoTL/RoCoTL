import os
from tensorboardX import SummaryWriter

def line_chart(xname, yname, xvalue, yalue, exp_result_dir, multiline=False, sametag=None):

    tensorboard_log_dir = os.path.join(exp_result_dir, f'tensorboard-log-{xname}_{yname}')
    os.makedirs(tensorboard_log_dir, exist_ok=True)    
    
    writer = SummaryWriter(log_dir = tensorboard_log_dir, comment= '-'+f'{xname}_{yname}')
    if multiline == True and sametag != None:
        writer.add_scalar(tag = sametag, scalar_value = yalue, global_step = xvalue)            # FP
    else:
        writer.add_scalar(tag = f'{xname}_{yname}', scalar_value = yalue, global_step = xvalue)   # FP+FN
    writer.close()      
   