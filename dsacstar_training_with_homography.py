import torch
import torch.optim as optim

import argparse
import time
import random
import dsacstar
import os

from network import Network
import datasets
from utils import tr, reverse_tr
import pickle
from torch.utils.tensorboard import SummaryWriter
import sys

print(f'system version\n{sys.version}')



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
parser.add_argument('--print_every', type=int, default=10, 
	help='print loss every this number of images')

parser.add_argument('--save_every', type=int, default=2, 
	help='save model every this number of epochs')


opt = parser.parse_args()

def compute_ABC(w_t_c, c_R_w, w_t_chat, chat_R_w, c_n, eye):
    """
    Computes A, B, and C matrix given estimated and ground truth poses
    and normal vector n.
    `w_t_c` and `w_t_chat` must have shape (batch_size, 3, 1).
    `c_R_w` and `chat_R_w` must have shape (batch_size, 3, 3).
    `n` must have shape (3, 1).
    `eye` is the (3, 3) identity matrix on the proper device.
    """
    chat_t_c = chat_R_w @ (w_t_c - w_t_chat)
#     print(f"in abc chatRW={chat_R_w.shape} and transpose={c_R_w.transpose(1,2).shape}")
    chat_R_c = chat_R_w @ c_R_w.transpose(1, 2)

    A = eye - chat_R_c
    C = c_n @ chat_t_c.transpose(1, 2)
    B = C @ A
    A = A @ A.transpose(1, 2)
    B = B + B.transpose(1, 2)
    C = C @ C.transpose(1, 2)

    return A, B, C


class LocalHomographyLoss(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        # `c_n` is the normal vector of the plane inducing the homographies in the ground-truth camera frame
        self.c_n = torch.tensor([0, 0, -1], dtype=torch.float32, device=device).view(3, 1)

        # `eye` is the (3, 3) identity matrix
        self.eye = torch.eye(3, device=device)

    def __call__(self, batch):
        A, B, C = compute_ABC(batch['w_t_c'], batch['c_R_w'], batch['w_t_chat'], batch['chat_R_w'], self.c_n, self.eye)

        xmin = batch['xmin'].view(-1, 1, 1)
        xmax = batch['xmax'].view(-1, 1, 1)
        B_weight = torch.log(xmax / xmin) / (xmax - xmin)
        C_weight = xmin * xmax

        error = A + B * B_weight + C / C_weight
        error = error.diagonal(dim1=1, dim2=2).sum(dim=1).mean()
        return error


if opt.dataset_name=='Cambridge':
    dataset = datasets.CambridgeDataset(f'homography_loss_function/datasets/Cambrige/{opt.scene_name}',opt.xmin_percentile, opt.xmax_percentile)
else:
    dataset = datasets.SevenScenesDataset(f'/mundus/mrahman527/projects/homography-loss-function/datasets/7-Scenes/{opt.scene_name}', opt.xmin_percentile, opt.xmax_percentile)




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


if with_init:
    writer_folder = 'with_init'
    checkpoint_folder = f'our_checkpoints/{opt.dataset_name}/{opt.scene_name}_with_init'
    os.makedirs(checkpoint_folder, exist_ok=True)

else:
    checkpoint_folder = f"our_checkpoints/{'7-Scenes'}/{'fire'}_without_init"
    if os.path.isdir(checkpoint_folder):
        checkpoint_folder = checkpoint_folder+'_1'

    os.makedirs(checkpoint_folder, exist_ok=True)

    writer_folder = 'without_init'
    
    
writer = SummaryWriter(os.path.join('logs',os.path.basename(os.path.normpath('7-Scenes')),'fire',writer_folder))


    

def train(network = network,trainset_loader=trainset_loader,testset_laoder=testset_loader,optimizer=optimizer, iteration=iteration, with_init=with_init, writer=writer,checkpoint_folder=checkpoint_folder):
    
    for epoch in range(iteration):
        
        network.train()
        print(f'epoch:{epoch}\n')
        print('========================train=====================')
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
            # print(f"cal loss for it {it}")
            # calculate loss
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
    
            print(f'epoch {epoch} iteration {it} loss train = {loss}')
            running_loss += loss
            # print(f'loss train = {running_loss}')

            torch.autograd.backward((scene_coordinates),(scene_coordinates_gradients.cuda()))
            optimizer.step()
            
            if it%opt.print_every==0 and it!=0:
                writer.add_scalar('train loss',running_loss/it,epoch*2000+it)
                print(f'logged it={it} train_loss={running_loss/it}')
        
        writer.add_scalar('per_epoch_training_loss',running_loss,epoch)

        if epoch%opt.save_every==0:
            checkpoint_path = os.path.join(checkpoint_folder,f'check_point_epoch_{epoch}.pt')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),

                }, checkpoint_path
            )
       
        print(f"after {epoch} epoch train loss: {running_loss}")
        print('========================test=====================')
        criterion = LocalHomographyLoss()
        #test
        network.eval()
        it = 0
        running_test_loss = 0        
        with torch.no_grad():
            for data in testset_loader:
                it+=1
                focal_length = data['K'][0][0][0]
                file = data['image_file']
                image = data['image'].cuda()
                wtc, crw = data['w_t_c'], data['c_R_w']

                # predict scene coordinates and neural guidance
                scene_coordinates = network(image)
                gt_pose = reverse_tr(crw, wtc)[0]
                out_pose = torch.zeros((4,4))
                # print('here')
                dsacstar.forward_rgb(
                    scene_coordinates.cpu(),
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    float(image.size(3)/2),
                    float(image.size(2)/2),
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE
                )
                # print('here 2')
                
                batch={}
                batch['w_t_c'] = data['w_t_c']
                batch['c_R_w'] = data['c_R_w']
                
                batch['w_t_chat'],batch['chat_R_w'] = tr(out_pose)
                batch['w_t_chat'] = batch['w_t_chat'].unsqueeze(0)
                batch['chat_R_w'] = batch['chat_R_w'].unsqueeze(0)
                batch['xmin'] = data['xmin']
                batch['xmax'] = data['xmax']
                
                # print('here 3')
                loss = criterion(batch)
                running_test_loss+=loss.item()
                
                if it%opt.print_every==0 and it!=0:
                    writer.add_scalar('test loss',running_loss/it,epoch*2000+it)
                    print(f"running test loss {running_test_loss/it}")
        
        writer.add_scalar('per_epoch_testing_loss',running_loss,epoch)
        print(f"after {epoch} epoch test loss:{running_loss}")

    

train(network = network, optimizer=optimizer,iteration=iteration)

