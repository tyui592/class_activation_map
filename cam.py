import json
import torch
import argparse

from utils import imload, imshow, imsave, array_to_cam, blend
from model import CAM

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int,
                    help='cpu: -1, gpu: 0 ~ n ', default=0)

    parser.add_argument('--network', type=str,
                    help='network to generate class activation map', default='resnet50')

    parser.add_argument('--image', type=str,
                    help='path of image', required=True)

    parser.add_argument('--topk', type=int,
                    help='make class activation maps of top k predicted classes', default=3)

    parser.add_argument('--imsize', type=int,
                    help='size to resize image (maintaining aspect ratio)', default=256)

    parser.add_argument('--cropsize', type=int,
                    help='size to crop center region ', default=None)

    parser.add_argument('--blend-alpha', type=float,
                    help='interpolate factor between input and cam', default=0.75)

    parser.add_argument('--save-path', type=str,
                    help='path of directory to save outputs', default='./')
    return parser

if __name__ == "__main__":
    # arguments
    parser = build_parser()
    args = parser.parse_args()

    # ImageNet class index to label 
    ## ref: https://discuss.pytorch.org/t/imagenet-classes/4923/2
    idx_to_label = json.load(open('imagenet_class_index.json'))
    idx_to_label = {int(key):value[1] for key, value in idx_to_label.items()}

    # set device
    device = torch.device('cuda:%d'%args.gpu_no if args.gpu_no >= 0 else 'cpu')
    network = CAM(args.network).to(device)
    network.eval()
    image = imload(args.image, args.imsize, args.cropsize).to(device)

    # make class activation map
    with torch.no_grad():
        prob, cls, cam = network(image, topk=args.topk)

        # tensor to pil image
        img_pil = imshow(image)
        img_pil.save(args.save_path+"input.jpg")

        for k in range(args.topk):
            print("Predict '%s' with %2.4f probability"%(idx_to_label[cls[k]], prob[k]))
            cam_ = cam[k].squeeze().cpu().data.numpy()
            cam_pil = array_to_cam(cam_)
            cam_pil.save(args.save_path+"cam_class__%s_prob__%2.4f.jpg"%(idx_to_label[cls[k]], prob[k]))

            # overlay image and class activation map
            blended_cam = blend(img_pil, cam_pil, args.blend_alpha)
            blended_cam.save(args.save_path+"blended_class__%s_prob__%2.4f.jpg"%(idx_to_label[cls[k]], prob[k]))
