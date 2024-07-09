
from __future__ import print_function
import cv2
import os.path
import numpy as np
import random
import torch
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
from criterion import CrossEntropyLoss
from dataset_ai2thor import ai2thor
import sys
sys.path.append("./sscdnet/correlation_package/build/lib.linux-x86_64-3.6")
import model
import wandb
import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score
import time
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


def soft_dice_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    soft_dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - soft_dice_loss)


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


class Training:
    def __init__(self, arguments):
        self.args = arguments
        self.valid_loss_score = 100
        self.not_save_count = 0
        self.last_save_file_name = ""
        self.savedir = self.args.checkpointdir

    def train(self):
        self.color_transform = Colorization(2)

        # wandb initialization
        if self.args.wandb is True:
            wandb.init(
                project=self.args.wandb_name,
                name=self.args.wandb_log_name,
                config=vars(self.args),
            )

        # Dataset loader for train, valid and test
        dataset_train = DataLoader(
            ai2thor(os.path.join(self.args.datadir, 'train')),
            num_workers=self.args.num_workers, batch_size=self.args.train_batch_size, shuffle=True)
        dataset_valid = DataLoader(
            ai2thor(os.path.join(self.args.datadir, 'valid')),
            num_workers=self.args.num_workers, batch_size=self.args.valid_batch_size, shuffle=False)

        self.test_path = os.path.join(self.savedir, 'test')
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        # Set loss function, optimizer and learning rate
        weight = torch.ones(2)
        criterion = CrossEntropyLoss(weight.cuda())
        optimizer = Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        lambda1 = lambda epoch: (float)(self.args.epochs + 1 - epoch) / (float)(self.args.epochs)
        model_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        epoch = 1

        while self.not_save_count < 5 and epoch <= 100:
            start_time = time.time()

            # Training loop
            self.model.train()
            train_epoch_loss = []
            train_miou = []
            train_f1_list = []
            for _, (inputs_train, mask_train) in enumerate(tqdm(dataset_train)):
                inputs_train = inputs_train.cuda()
                mask_train = mask_train.cuda()
                inputs_train = Variable(inputs_train)
                mask_train = Variable(mask_train)

                optimizer.zero_grad()
                outputs_train = self.model(inputs_train)
                self.loss = criterion(outputs_train, mask_train[:, 0])
                self.loss += soft_dice_loss(mask_train, outputs_train)

                # save train loss
                train_epoch_loss.append(self.loss.item())

                self.loss.backward()
                optimizer.step()

                train_iou = IoU(outputs_train, mask_train[:, 0])
                train_miou.extend(train_iou)
                
                train_f1 = macro_f1(outputs_train, mask_train[:, 0])
                train_f1_list.extend(train_f1)

            # validation
            self.model.eval()
            valid_epoch_loss = []
            valid_miou = []
            valid_f1_list = []
            for _, (inputs_valid, mask_valid) in enumerate(tqdm(dataset_valid)):
                inputs_valid = inputs_valid.cuda()
                mask_valid = mask_valid.cuda()
                inputs_valid = Variable(inputs_valid)
                mask_valid = Variable(mask_valid)

                with torch.no_grad():
                    outputs_valid = self.model(inputs_valid)
                    self.loss = criterion(outputs_valid, mask_valid[:, 0])
                    self.loss += soft_dice_loss(mask_valid, outputs_valid)

                # save valid loss
                valid_epoch_loss.append(self.loss.item())

                valid_iou = IoU(outputs_valid, mask_valid[:, 0])
                valid_miou.extend(valid_iou)

                valid_f1 = macro_f1(outputs_valid, mask_valid[:, 0])
                valid_f1_list.extend(valid_f1)

            model_lr_scheduler.step()

            end_time = time.time()
            all_secs = int(end_time - start_time)
            mins = all_secs // 60
            secs = all_secs % 60

            train_loss = sum(train_epoch_loss) / len(train_epoch_loss)
            valid_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)
            train_miou = np.mean(np.array(train_miou))
            train_f1 = np.mean(np.array(train_f1_list))
            valid_miou = np.mean(np.array(valid_miou))
            valid_f1 = np.mean(np.array(valid_f1_list))

            # print time
            print(
                "----------- epoch : %d / %d" % (epoch, self.args.epochs),
                " | time in %d minutes, %d seconds -----------\n" % (mins, secs),
            )
            print("## loss ##\n")
            print("train : {:.4f} | valid : {:.4f}\n".format(train_loss, valid_loss))
            print("## mIoU ##\n")
            print("train : {:.4f} | valid : {:.4f}\n".format(train_miou, valid_miou))
            print("## F1 ##\n")
            print("train : {:.4f} | valid : {:.4f}\n".format(train_f1, valid_f1))
            

            # wandb
            if self.args.wandb == True:
                output = {
                    "train_loss":train_loss, 
                    "train_miou":train_miou,
                    "train_macrof1":train_f1,
                    "valid_loss":valid_loss, 
                    "valid_miou":valid_miou,
                    "valid_macrof1":valid_f1,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(output, step=epoch)

            # save the model if it's the best
            if self.valid_loss_score >= valid_loss:
                self.not_save_count = 0
                self.valid_loss_score = valid_loss
                print(f"saving model file (epoch: {epoch})")
                self.checkpoint(epoch)
            else:
                self.not_save_count += 1
            
            epoch += 1

        test_model = model.Model(inc=6, outc=2)
        fn_model = self.last_save_file_name
        test_model.load_state_dict(torch.load(fn_model))
        test_model = test_model.cuda()
        self.test(test_model)

    def test(self, model):
        print("#### Test mode ####")
        model.eval()
        if self.args.image == True:
            save_dir = os.path.join(self.savedir, "output", str(datetime.datetime.now()))
            os.makedirs(save_dir, exist_ok=True)
        dataset_test = DataLoader(
            ai2thor(os.path.join(self.args.datadir, 'test')),
            num_workers=self.args.num_workers, batch_size=self.args.valid_batch_size, shuffle=False)
        test_miou = []
        test_f1_list = []
        test_count = 0
        with torch.no_grad():
            for _, (inputs_test, mask_test) in enumerate(tqdm(dataset_test)):
                inputs_test = inputs_test.cuda()
                mask_test = mask_test.cuda()
                inputs_test = Variable(inputs_test)
                mask_test = Variable(mask_test)

                outputs_test = model(inputs_test)

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


    def checkpoint(self, epoch):
        filename = '{0:08d}-{1}.pth'.format(epoch, datetime.datetime.now())

        torch.save(self.model.state_dict(), os.path.join(self.savedir, filename))
        print('save: {0} (epoch: {1})\n'.format(filename, epoch))
        if os.path.exists(self.last_save_file_name):
            os.remove(self.last_save_file_name)
        self.last_save_file_name = os.path.join(self.savedir, filename)


    def run(self):
        print('Co-Scale Cross-Attentional Transformer')
        self.model = model.Model(inc=6, outc=2)

        self.model = self.model.cuda()
        self.train()


if __name__ == '__main__':

    opts = main_argparse()
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    training = Training(opts)
    training.run()
