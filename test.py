import numpy as np
import cv2
import os.path
import torch
from torch.autograd import Variable
import sys
sys.path.append("./sscdnet/correlation_package/build/lib.linux-x86_64-3.6")
import model
from torch.utils.data import DataLoader
from dataset_ai2thor import ai2thor
import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score
from arguments import main_argparse


def colormap():
    cmap=np.zeros([2, 3]).astype(np.uint8)

    cmap[0,:] = np.array([0, 0, 0])
    cmap[1,:] = np.array([255, 255, 255])

    return cmap


def IoU(pred:torch.Tensor, gt:torch.Tensor):
    pred = pred.argmax(1)
    all_iou = []

    all_mul = torch.mul(pred, gt)
    all_add = torch.add(pred, gt)
    for i in range(pred.shape[0]):
        intersection = torch.sum(all_mul[i])
        union = torch.sum(all_add[i]) - intersection
        if intersection == 0 or union == 0:
            iou = 0
        else:
            iou = float(intersection) / float(union)
        all_iou.append(iou)

    return all_iou


def macro_f1(pred:torch.Tensor, true:torch.Tensor) -> float:
    pred = pred.argmax(1)
    all_score = []

    for i in range(pred.shape[0]):
        score = f1_score(
            true[i].to('cpu').detach().numpy().copy().ravel(),
            pred[i].to('cpu').detach().numpy().copy().ravel(),
            average='macro'
        )
        all_score.append(score)

    return all_score


class Colorization:

    def __init__(self, n=2):
        self.cmap = colormap()
        self.cmap = torch.from_numpy(np.array(self.cmap[:n]))

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


class Test:
    def __init__(self, arguments):
        self.args = arguments
        self.savedir = self.args.checkpointdir

    def test(self):
        self.color_transform = Colorization(2)

        print("#### Test mode ####")
        self.model.eval()
        if self.args.image == True:
            save_dir = os.path.join(self.savedir, "output", str(datetime.datetime.now()))
            os.makedirs(save_dir, exist_ok=True)
        dataset_test = DataLoader(
            ai2thor(os.path.join(self.args.datadir, 'test')),
            batch_size=self.args.batch_size, shuffle=False)
        test_miou = []
        test_f1_list = []
        test_count = 0
        with torch.no_grad():
            for step, (inputs_test, mask_test) in enumerate(tqdm(dataset_test)):
                inputs_test = inputs_test.cuda()
                mask_test = mask_test.cuda()
                inputs_test = Variable(inputs_test)
                mask_test = Variable(mask_test)

                outputs_test = self.model(inputs_test)

                if self.args.image == True:
                    for i in range(inputs_test.size()[0]):
                        inputs = inputs_test[i].cpu().data
                        t0_test = inputs[0:3, :, :]
                        t1_test = inputs[3:6, :, :]
                        t0_test = (t0_test + 1.0) * 128
                        t1_test = (t1_test + 1.0) * 128
                        mask_gt = mask_test[i].to('cpu').detach().numpy().copy().astype(np.uint8) * 255

                        outputs = outputs_test[i][np.newaxis, :, :, :]
                        outputs = outputs[:, 0:2, :, :]
                        mask_pred = np.transpose(self.color_transform(outputs[0].cpu().max(0)[1][np.newaxis, :, :].data).numpy(), (1, 2, 0)).astype(np.uint8)

                        img_full = self.save_results(t0_test, t1_test, mask_pred, mask_gt)
                        cv2.imwrite(os.path.join(save_dir, f"{test_count:05d}_full.png"), img_full)
                        cv2.imwrite(os.path.join(save_dir, f"{test_count:05d}_mask.png"), mask_pred)
                        test_count += 1

                test_iou = IoU(outputs_test, mask_test[:, 0])
                test_miou.extend(test_iou)

                test_f1 = macro_f1(outputs_test, mask_test[:, 0])
                test_f1_list.extend(test_f1)

        test_miou = np.mean(np.array(test_miou))
        test_f1 = np.mean(np.array(test_f1_list))

        print("## mIoU ##\n")
        print("test : {:.4f}\n".format(test_miou))
        print("## F1 ##\n")
        print("test : {:.4f}\n".format(test_f1))


    def save_results(self, t0, t1, mask_pred, mask_gt):

        rows = cols = 256
        img_out = np.zeros((rows * 2, cols * 2, 3), dtype=np.uint8)
        img_out[0:rows, 0:cols, :] = np.transpose(t0.numpy(), (1, 2, 0)).astype(np.uint8)
        img_out[0:rows, cols:cols * 2, :] = np.transpose(t1.numpy(), (1, 2, 0)).astype(np.uint8)
        img_out[rows:rows * 2, 0:cols, :] = cv2.cvtColor(np.transpose(mask_gt, (1, 2, 0)), cv2.COLOR_GRAY2RGB)
        img_out[rows:rows * 2, cols:cols * 2, :] = mask_pred

        return img_out


    def run(self):
        print('Co-Scale Cross-Attentional Transformer')
        self.model = model.Model(inc=6, outc=2)
        fn_model = os.path.join(self.args.checkpointdir, self.args.model)

            
        if os.path.isfile(fn_model) is False:
            print("Error: Cannot read file ... " + fn_model)
            exit(-1)
        else:
            print("Reading model ... " + fn_model)
        
        self.model.load_state_dict(torch.load(fn_model))
        self.model = self.model.cuda()
        self.test()


if __name__ == '__main__':

    opts = main_argparse()
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    test = Test(opts)
    test.run()


