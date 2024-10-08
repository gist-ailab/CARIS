import torch
import torch.utils.data
from torch import nn

from model import builder
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import functional as FC

import pickle
from bert.tokenization_bert import BertTokenizer


class Resize(object):
    def __init__(self, h, w, eval_mode=False):
        self.h = h
        self.w = w
        self.eval_mode = eval_mode

    def __call__(self, image, target):
        image = FC.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        if not self.eval_mode:
            if isinstance(target, list):
                target_new = []
                for _target in target:
                    target_new.append(F.resize(_target, (self.h, self.w), interpolation=F.InterpolationMode.NEAREST))
                target = target_new
            else:
                pass    ## only inference_demo.py
                # target = F.resize(target, (self.h, self.w), interpolation=F.InterpolationMode.NEAREST)
        return image, target


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDatasetTest
    ds = ReferDatasetTest(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes

def batch_IoU(pred, gt):
    intersection = torch.logical_and(pred, gt).sum(1)
    union = torch.logical_or(pred, gt).sum(1)
    # intersection = torch.sum(torch.mul(pred, gt), dim=1)
    # union = torch.sum(torch.add(pred, gt), dim=1) - intersection

    iou = intersection.float() / union.float()

    return iou, intersection, union

def batch_evaluate(model, data):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_num = 8
    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()
    eval_seg_iou_list = [.5, .7, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()

    ## tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    ## visualize
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    def normalize(image):
        return (image - image.min()) / (image.max() - image.min())
    import os
    os.makedirs(f'./output/demo', exist_ok=True)

    image = data['image']
    targets = data['targets']
    sentences = data['sentences']
    attentions = data['attentions']

    with torch.no_grad():
        # image, targets, sentences, attentions = data
        image, sentences, attentions = image.cuda(non_blocking=True),\
                                                sentences.cuda(non_blocking=True),\
                                                attentions.cuda(non_blocking=True)
        target = targets['mask'].cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        output = model(image, sentences, l_mask=attentions)

        iou, I, U = batch_IoU(output.flatten(1), target.flatten(1))
        acc_ious += iou.sum()
        cum_I += I.sum()
        cum_U += U.sum()
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (iou >= eval_seg_iou).sum()

        sentences_raw = []
        for sentence in sentences:
            # decoded_sentence = tokenizer.decode(sentence[0], skip_special_tokens=True)
            decoded_sentence = tokenizer.decode(sentence.view(-1).tolist(), skip_special_tokens=True)
            sentences_raw.append(decoded_sentence)
            # print(decoded_sentence)

        idx = 0
        # for idx in range(image.shape[0]):
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(normalize(image[idx].permute(1, 2, 0).cpu().numpy()))
        # plt.imshow(to_pil_image(image[idx]))
        plt.axis('off')
        plt.title('input', fontdict={'fontsize' : 10})

        plt.subplot(1, 2, 2)
        plt.imshow(normalize(image[idx].permute(1, 2, 0).cpu().numpy()))
        plt.imshow(output[idx][0].cpu(), alpha=0.5)
        plt.axis('off')
        plt.title('pred: CARIS', fontdict={'fontsize' : 10})

        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=2.4)
        plt.suptitle(f'"{sentences_raw[idx]}"', fontsize=12)
        # if iou[idx] < 0.5:
        #     plt.text(0.5, 0.94, f'iou: {iou[idx]:.2f}', fontsize=15, color='red', ha='center', va='top', transform=plt.gcf().transFigure)
        # else:
        #     plt.text(0.5, 0.94, f'iou: {iou[idx]:.2f}', fontsize=15, color='blue', ha='center', va='top', transform=plt.gcf().transFigure)
        plt.savefig(f'./output/demo/{idx}.png', bbox_inches='tight')
        
            
    torch.cuda.synchronize()
    cum_I = cum_I.cpu().numpy()
    cum_U = cum_U.cpu().numpy()
    acc_ious = acc_ious.cpu().numpy()
    seg_correct = seg_correct.cpu().numpy()

    mIoU = acc_ious / total_num
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / total_num)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

def get_transform(args):
    transforms = [Resize(args.img_size, args.img_size, eval_mode=args.eval_ori_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg),axis=1)
    U = np.sum(np.logical_or(pred_seg, gd_seg),axis=1)

    return I, U

def main(args):
    device = torch.device(args.device)
    # dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    # print(len(dataset_test))
    # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8,
    #                                                sampler=test_sampler, num_workers=args.workers)
    with open(args.pickle_path, 'rb') as file:
        data = pickle.load(file)
    
    
    ##  custom image input  ##
    new_img = Image.open(args.image_path).convert("RGB")
    dummy = torch.zeros((args.img_size, args.img_size), dtype=torch.int64)
    transform = get_transform(args=args)
    new_img_tensor, _ = transform(new_img, dummy)

    data['image'][0] = data['image'][1] = new_img_tensor
    
    ##  custom text input  ##
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    
    input_text = args.input_text
    encoded_inputs = tokenizer.encode_plus(
        input_text,  # The input text to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        max_length=20,  # Maximum length of the tokenized input (from your tensor example)
        pad_to_max_length=True,  # Pad the input to the max_length with [PAD] token
        return_tensors='pt',  # Return tensors
        return_attention_mask=True  # Return attention mask
    )
    
    data['sentences'][0] = encoded_inputs['input_ids']
    data['attentions'][0] = encoded_inputs['attention_mask']

    print(args.model)
    single_model = builder.__dict__[args.model](pretrained='',args=args)
    utils.load_model(single_model, args.resume)
    model = single_model.to(device)

    batch_evaluate(model, data)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    
    ##########   for debug   ##########
    args.model = "caris"
    args.swin_type = "base"
    args.dataset = "refcoco"
    args.split = "val"
    args.img_size = 448
    args.resume = "/SSDe/heeseon_rho/src/CARIS/output/model_best_refcoco.pth"
    args.bert_tokenizer = "bert-base-uncased"
    args.ck_bert = "/SSDe/heeseon_rho/src/CARIS/ckpt/bert-base-uncased/"
    args.refer_data_root = "/ailab_mat/dataset/refCOCO/images"
    args.refer_root = "/ailab_mat/dataset/RIS"
    args.pickle_path = "/SSDe/heeseon_rho/src/CARIS/output/visualize/473.pickle"
    args.image_path = "/SSDe/heeseon_rho/src/CARIS/input/hyundai.png"
    args.input_text = "car door"
    ###################################

    print('Image size: {}'.format(str(args.img_size)))
    if args.eval_ori_size:
        print('Eval mode: original')
    else:
        print('Eval mode: resized')
    main(args)
