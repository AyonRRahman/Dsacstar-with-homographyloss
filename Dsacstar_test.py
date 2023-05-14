import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

import time
import math

import argparse

import dsacstar
from network import Network
import datasets
from utils import tr, reverse_tr
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(
    description='Test a network on a specific scene',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('dataset_name',help='name of the dataset. e.g Cambridge,7-Scenes')
parser.add_argument('scene_name', help='name of the scene. e.g chess, fire, ShopFacade')
parser.add_argument('--xmin_percentile', help='xmin depth percentile', type=float, default=0.025)
parser.add_argument('--xmax_percentile', help='xmax depth percentile', type=float, default=0.975)

parser.add_argument('--hypotheses','-hyps', help='number of hypotheses', type=int, default=64)
parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--tiny', '-tiny', action='store_true',
	help='Train a model with massively reduced capacity for a low memory footprint.')

parser.add_argument('--path_to_checkpoints' , type=str, default=100, 
	help='path to the checkpoint folder')

parser.add_argument('--run' , type=str, 
	help='fplder name for tensorboard')

opt = parser.parse_args()

if not os.path.isdir(opt.path_to_checkpoints):
    raise Exception(" The checkpoint path doesnot exist. Please give the correct path.")


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




criterion = LocalHomographyLoss()
if opt.dataset_name=='Cambridge':
    dataset = datasets.CambridgeDataset(f'homography_loss_function/datasets/Cambrige/{opt.scene_name}',opt.xmin_percentile, opt.xmax_percentile)
else:
    dataset = datasets.SevenScenesDataset(f'/mundus/mrahman527/projects/homography-loss-function/datasets/7-Scenes/{opt.scene_name}', opt.xmin_percentile, opt.xmax_percentile)


writer_folder = f'test/{str(opt.run)}'
writer = SummaryWriter(os.path.join('logs',os.path.basename(os.path.normpath('7-Scenes')),'fire',writer_folder))


train_dataset = datasets.RelocDataset(dataset.train_data)
test_dataset = datasets.RelocDataset(dataset.test_data)

trainset_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, num_workers=6, batch_size=1)
testset_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=6, batch_size=1)

#load network
network = Network(torch.zeros((3)), opt.tiny)
network.cuda()
check_path = opt.path_to_checkpoints

check_point_list = os.listdir(check_path)

for checkpoint in check_point_list:
    epoch = int(checkpoint[:-3].split('_')[-1])
    print(f'testing for epoch {epoch}')
    
    check_point = torch.load(os.path.join(check_path, checkpoint))
    network.load_state_dict(check_point['model_state_dict'])
    network.eval()

    running_homography_loss = 0

    it = 0
    rErrs = []
    tErrs = []
    avg_time = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    with torch.no_grad():
        for data in testset_loader:
            it+=1
            focal_length = data['K'][0][0][0]
            image = data['image'].cuda()
            
            wtc, crw = data['w_t_c'], data['c_R_w']

            start_time = time.time()
            scene_coordinates = network(image)
            scene_coordinates = scene_coordinates.cpu()
            gt_pose = reverse_tr(crw, wtc)[0]
            out_pose = torch.zeros((4, 4))
            dsacstar.forward_rgb(
				scene_coordinates, 
				out_pose, 
				opt.hypotheses, 
				opt.threshold,
				focal_length, 
				float(image.size(3) / 2), #principal point assumed in image center
				float(image.size(2) / 2), 
				opt.inlieralpha,
				opt.maxpixelerror,
				network.OUTPUT_SUBSAMPLE)

            
            avg_time += time.time()-start_time

            #homography loss calc
            batch={}
            batch['w_t_c'] = data['w_t_c']
            batch['c_R_w'] = data['c_R_w']
            
            batch['w_t_chat'],batch['chat_R_w'] = tr(out_pose)
            batch['w_t_chat'] = batch['w_t_chat'].unsqueeze(0)
            batch['chat_R_w'] = batch['chat_R_w'].unsqueeze(0)
            batch['xmin'] = data['xmin']
            batch['xmax'] = data['xmax']

            loss = criterion(batch)
            running_homography_loss+=loss.item()

            # calculate pose errors
            t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))
            # print(f"t err {t_err}")
            gt_R = gt_pose[0:3,0:3].numpy()
            out_R = out_pose[0:3,0:3].numpy()

            r_err = np.matmul(out_R, np.transpose(gt_R))
            r_err = cv2.Rodrigues(r_err)[0]
            r_err = np.linalg.norm(r_err) * 180 / math.pi
            # print(f"r err {r_err}")

            # print(gt_pose)
            # print(out_pose)


            # print("\nRotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err*100))

            rErrs.append(r_err)
            tErrs.append(t_err * 100)

            if r_err < 5 and t_err < 0.05:
                pct5 += 1
            if r_err < 2 and t_err < 0.02:
                pct2 += 1
            if r_err < 1 and t_err < 0.01:
                pct1 += 1

            
    median_idx = int(len(rErrs)/2)
    tErrs.sort()
    rErrs.sort()
    avg_time /= len(rErrs)

    print("\n===================================================")
    print("\nTest complete.")
    writer.add_scalar('avg Homography loss', running_homography_loss/len(rErrs),epoch)

    print('\nAccuracy:')
    print('\n5cm5deg: %.1f%%' %(pct5 / len(rErrs) * 100))
    writer.add_scalar('5cm5deg',pct5 / len(rErrs) * 100,epoch)

    print('2cm2deg: %.1f%%' % (pct2 / len(rErrs) * 100))
    writer.add_scalar('2cm2deg',pct2 / len(rErrs) * 100, epoch)
    
    print('1cm1deg: %.1f%%' % (pct1 / len(rErrs) * 100))
    writer.add_scalar('1cm1deg',pct1 / len(rErrs) * 100, epoch)

    print("\nMedian Error: %.1fdeg, %.1fcm" % (rErrs[median_idx], tErrs[median_idx]))
    writer.add_scalar('Median rotation Error',rErrs[median_idx], epoch)
    writer.add_scalar('Median translation Error',tErrs[median_idx], epoch)
    writer.add_scalar('avg rotation Error',np.mean(np.array(rErrs)), epoch)
    writer.add_scalar('avg translation Error',np.mean(np.array(tErrs)), epoch)

    
    
    print("Avg. processing time: %4.1fms" % (avg_time * 1000))
    writer.add_scalar('Avg. processing time(ms)',avg_time * 1000, epoch)

    

    
