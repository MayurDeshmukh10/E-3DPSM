# from model import EgoHPE
from EventEgoPoseEstimation.model.network import EgoHPE
# from settings import config
from configs.settings import config
import torch
import time
from ptflops import get_model_complexity_info

image_size = config.MODEL.IMAGE_SIZE
image = torch.zeros(1, 1, 2, image_size[0], image_size[1]).cuda()


net = EgoHPE(config, num_joints=16,
             eros=True,
            input_channel=2,
            posenet_input_channel=2,
            image_size=[256, 192],
            batch_size=1
        )

prev_s5_states = {
            0: None,
            1: None,
            2: None,
            3: None
        }
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(net, (2, 2, image_size[0], image_size[1]), as_strings=True, print_per_layer_stat=True, verbose=True)
    
    import pdb; pdb.set_trace()
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('FLOPS complexity: ', 2 * macs))

net = torch.compile(net)
net = net.cuda().eval()

# net = net.half()
# image = image.half()    

count = 0
avg_time = 0
with torch.no_grad():
    while True:
        t = time.time()
        net(image)
        avg_time += time.time() - t

        count += 1  
                
        if count % 1000 == 0:
            print(f'avg time: {avg_time / count * 1000} fps: {1 / (avg_time / count)}')
            avg_time = 0
            count = 0