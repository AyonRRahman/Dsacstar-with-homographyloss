import torch
import torch.optim as optim

import argparse
import time
import random
import dsacstar_with_homography
import os

from network import Network
from homography_loss_function import datasets
from utils import tr, reverse_tr
import pickle

parser = argparse.ArgumentParser(
    description='Train a network on a specific scene',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('dataset_name',help='name of the dataset. e.g Cambridge,7-Scenes')
parser.add_argument('scene_name', help='name of the scene. e.g chess, fire, ShopFacade')

parser.add_argument('--xmin_percentile', help='xmin depth percentile', type=float, default=0.025)
parser.add_argument('--xmax_percentile', help='xmax depth percentile', type=float, default=0.975)
parser.add_argument('--hypotheses','-hyps', help='number of hypotheses', type=int, default=64)

parser.add_argument('--network_in', help='file name of a network initialized for the scene', type=str, default='None')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
	help='learning rate')

parser.add_argument('--iterations', '-it', type=int, default=100000, 
	help='number of training iterations, i.e. network parameter updates')

parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
	help='weight of rotation part of pose loss')

parser.add_argument('--weighttrans', '-wt', type=float, default=100.0, 
	help='weight of translation part of pose loss')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
	help='robust square root loss after this threshold')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--tiny', '-tiny', action='store_true',
	help='Train a model with massively reduced capacity for a low memory footprint.')
parser.add_argument('--print_every', type=int, default=100, 
	help='print loss every this number of images')

parser.add_argument('--save_every', type=int, default=10, 
	help='save model every this number of epochs')


opt = parser.parse_args()

if opt.dataset_name=='Cambridge':
    dataset = datasets.CambridgeDataset(f'homography_loss_function/datasets/Cambrige/{opt.scene_name}',opt.xmin_percentile, opt.xmax_percentile)
else:
    dataset = datasets.SevenScenesDataset(f'homography_loss_function/datasets/7-Scenes/{opt.scene_name}', opt.xmin_percentile, opt.xmax_percentile)




train_dataset = datasets.RelocDataset(dataset.train_data)
test_dataset = datasets.RelocDataset(dataset.test_data)

trainset_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, num_workers=6, batch_size=1)
testset_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=6, batch_size=1)

# load network
network = Network(torch.zeros((3)), opt.tiny)
with_init=False
if opt.network_in!='None':
    network.load_state_dict(torch.load(opt.network_in))
    with_init=True
network = network.cuda()
network.train()


optimizer = torch.optim.Adam(network.parameters(),lr=opt.learningrate)
iteration = opt.iterations

def train(network = network, optimizer=optimizer, criterion=criterion, iteration=iteration, with_init=with_init):
    if with_init:
        checkpoint_folder = f'our_checkpoints/{opt.dataset_name}/{opt.scene_name}_with_init'
        os.mkdir(checkpoint_folder)
    else:
        checkpoint_folder = f'our_checkpoints/{opt.dataset_name}/{opt.scene_name}_without_init'
        if os.path.isdir(checkpoint_folder):
            checkpoint_folder = checkpoint_folder+'_1'
        
        os.mkdir(checkpoint_folder)
    
    loss_list = []
    per_epoch_loss_list = []
    for epoch in range(iteration):
        print(f'epoch:{epoch}\n')
        running_loss = 0
        it = 0
        for data in trainset_loader:
            it+=1
            with torch.no_grad():
                optimizer.zero_grad()

            focal_length = data['K'][0][0][0]
            file = data['image_file']
            image = data['image'].cuda()
            start_time = time.time()
            wtc, crw = data['w_t_c'], data['c_R_w']
            
            # predict scene coordinates and neural guidance
            scene_coordinates = network(image)
            scene_coordinates_gradients = torch.zeros(scene_coordinates.size())
            gt_pose = reverse_tr(crw, wtc)[0]
            # print(f"shape pose={gt_pose.shape}")
            # print(f"xmin = {data['xmin']} shape {data['xmin'].shape}")
            # print(f"xmax = {data['xmax']} shape {data['xmax'].shape}")

            # pose from RGB
            loss = dsacstar.backward_rgb(
                scene_coordinates.cpu(),
                scene_coordinates_gradients,
                gt_pose, 
                opt.hypotheses, 
                opt.threshold,
                focal_length, 
                float(image.size(3) / 2), #principal point assumed in image center
                float(image.size(2) / 2),
                opt.weightrot,
                opt.weighttrans,
                opt.softclamp,
                opt.inlieralpha,
                opt.maxpixelerror,
                network.OUTPUT_SUBSAMPLE,
                random.randint(0,1000000), #used to initialize random number generator in cpp
                data['xmin'].item(),
                data['xmax'].item()
            )
    
            
            running_loss += loss
            print(f"Done 1 batch scene coor grads {scene_coordinates_gradients}, loss = {loss}")
            torch.autograd.backward((scene_coordinates),(scene_coordinates_gradients.cuda()))
            optimizer.step()
            if it%opt.print_every==0 and it!=0:
                loss_list.append(running_loss/it)
                print(f'it={it}')
        
        per_epoch_loss_list.append(running_loss)

        if epoch%opt.save_every==0:
            checkpoint_path = os.path.join(checkpoint_folder,f'check_point_epoch_{epoch}.pt')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),

                }, checkpoint_path
            )
       
        print(f"loss: {running_loss}")
    
    with open(os.path.join(checkpoint_folder,'loss_list.pickle'),'wb') as f:
        pickle.dump({'loss_list':loss_list,'per_epoch_loss_list':per_epoch_loss_list}, f)

    

train(network = network, optimizer=optimizer, criterion=criterion, iteration=iteration)

