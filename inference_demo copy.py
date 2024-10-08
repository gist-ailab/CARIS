import torch
import torch.utils.data
from torch import nn
import numpy as np
from PIL import Image
import torch.nn.functional as F

from model import builder
import utils
import transforms as T

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    # dummy_target = torch.zeros((image.size[0], image.size[1]), dtype=torch.int64)
    dummy_target = torch.zeros((args.img_size, args.img_size), dtype=torch.int64)

    image, _ = transform(image, dummy_target)
    return image

def batch_IoU(pred, gt):
    intersection = torch.logical_and(pred, gt).sum(1)
    union = torch.logical_or(pred, gt).sum(1)
    iou = intersection.float() / union.float()
    return iou, intersection, union

def infer_on_custom_image(model, image_path, sentence, tokenizer, transform):
    model.eval()
    image = load_image(image_path, transform).unsqueeze(0).cuda()

    # Tokenize sentence
    sentence_tokenized = tokenizer.encode(sentence, return_tensors='pt').cuda()

    with torch.no_grad():
        output = model(image, sentence_tokenized)

        # Post-processing
        output = output.squeeze(0).cpu().numpy()
        output = np.argmax(output, axis=0)  # assuming output is in the shape [num_classes, H, W]

        return output

def visualize_results(image_path, output, sentence):
    image = Image.open(image_path).convert("RGB")
    output = Image.fromarray(output.astype(np.uint8) * 255)

    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(output, alpha=0.5)
    plt.title(f"Predicted Mask: {sentence}")
    plt.axis('off')

    plt.show()

def get_transform(args):
    transforms = [
        T.Resize(args.img_size, args.img_size, eval_mode=args.eval_ori_size),   # height, width 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)

def main(args):
    device = torch.device(args.device)

    # Load the model
    single_model = builder.__dict__[args.model](pretrained='', args=args)
    utils.load_model(single_model, args.resume)
    model = single_model.to(device)

    # Load tokenizer
    from bert.tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    # Set up transformations
    transform = get_transform(args)

    # Custom image and sentence inputs
    custom_image_path = "/SSDe/heeseon_rho/src/CARIS/input/platter.png"  # replace with your image path
    custom_sentence = "middle platter"  # replace with your sentence

    # Perform inference
    output = infer_on_custom_image(model, custom_image_path, custom_sentence, tokenizer, transform)

    # Visualize the results
    visualize_results(custom_image_path, output, custom_sentence)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    
    ##########   for debug   ##########
    args.model = "caris"
    args.swin_type = "base"
    args.dataset = "refcoco"
    # args.split = "val"
    args.img_size = 448
    args.resume = "/SSDe/heeseon_rho/src/CARIS/output/model_best_refcoco.pth"
    args.bert_tokenizer = "bert-base-uncased"
    args.ck_bert = "/SSDe/heeseon_rho/src/CARIS/ckpt/bert-base-uncased/"
    # args.refer_data_root = "/ailab_mat/dataset/refCOCO/images"
    # args.refer_root = "/ailab_mat/dataset/RIS"
    ###################################

    print('Image size: {}'.format(str(args.img_size)))
    if args.eval_ori_size:
        print('Eval mode: original')
    else:
        print('Eval mode: resized')
    main(args)
