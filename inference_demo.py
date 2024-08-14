import torch
import torch.utils.data
from torch import nn

from model import builder
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F


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
    from bert.tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    ## visualize
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    def normalize(image):
        return (image - image.min()) / (image.max() - image.min())
    import os
    os.makedirs(f'./output/demo', exist_ok=True)
    total_idx = 0

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

        for idx in range(image.shape[0]):
            plt.figure(figsize=(20, 10))

            plt.subplot(1, 3, 1)
            plt.imshow(normalize(image[idx].permute(1, 2, 0).cpu().numpy()))
            # plt.imshow(to_pil_image(image[idx]))
            plt.axis('off')
            plt.title('input', fontdict={'fontsize' : 20})

            plt.subplot(1, 3, 2)
            plt.imshow(normalize(image[idx].permute(1, 2, 0).cpu().numpy()))
            plt.imshow(targets['mask'][idx], alpha=0.5)
            plt.axis('off')
            plt.title('ground truth', fontdict={'fontsize' : 20})

            plt.subplot(1, 3, 3)
            plt.imshow(normalize(image[idx].permute(1, 2, 0).cpu().numpy()))
            plt.imshow(output[idx][0].cpu(), alpha=0.5)
            plt.axis('off')
            plt.title('pred: CARIS', fontdict={'fontsize' : 20})

            plt.subplots_adjust(wspace=0.05, hspace=0.05, top=2.4)
            plt.suptitle(f'"{sentences_raw[idx]}"', fontsize=26)
            if iou[idx] < 0.5:
                plt.text(0.5, 0.94, f'iou: {iou[idx]:.2f}', fontsize=24, color='red', ha='center', va='top', transform=plt.gcf().transFigure)
                plt.savefig(f'./output/{args.dataset}/u50/{total_idx}.png', bbox_inches='tight')
            else:
                plt.text(0.5, 0.94, f'iou: {iou[idx]:.2f}', fontsize=24, color='blue', ha='center', va='top', transform=plt.gcf().transFigure)
            plt.savefig(f'./output/{args.dataset}/{total_idx}.png', bbox_inches='tight')
            total_idx += 1

            
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
    transforms = [T.Resize(args.img_size, args.img_size, eval_mode=args.eval_ori_size),
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
    
    # replace data[1] image to platter1.png
    


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
    args.resume = "/SSDe/heeseon/src/CARIS/output/model_best_refcoco.pth"
    args.bert_tokenizer = "bert-base-uncased"
    args.ck_bert = "/SSDe/heeseon/src/CARIS/ckpt/bert-base-uncased/"
    args.refer_data_root = "/ailab_mat/dataset/refCOCO/images"
    args.refer_root = "/ailab_mat/dataset/RIS"
    args.pickle_path = "/SSDe/heeseon/src/CARIS/output/visualize/473.pickle"
    args.image_path = "/SSDe/heeseon/src/CARIS/input/platter1.png"
    ###################################

    print('Image size: {}'.format(str(args.img_size)))
    if args.eval_ori_size:
        print('Eval mode: original')
    else:
        print('Eval mode: resized')
    main(args)
