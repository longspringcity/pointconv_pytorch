import argparse
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from tqdm import tqdm

import provider
from data_utils.TranslationDataLoader import TranslationDataLoader
from model.pointconv_trans import PointConvDensityTrans as PointConvTrans
from utils.utils import trans_test, save_checkpoint


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=5, help='batch size in training')
    # parser.add_argument('--batchsize', type=int, default=2, help='batch size in training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    # parser.add_argument('--num_workers', type=int, default=0, help='Worker Number [default: 16]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default='pretrained/pointconv_modelnet40-0.892500-0059.pth',
                        help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    # parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--model_name', default='pointconv_modelnet40', help='model name')
    # parser.add_argument('--normal', action='store_true', default=False,
    #                     help='Whether to use normal information [default: False]')
    parser.add_argument('--normal', default=False, help='Whether to use normal information')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-' % args.model_name + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        '---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = './data/point_data/'

    TRAIN_DATASET = TranslationDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                          normal_channel=args.normal)
    TEST_DATASET = TranslationDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                         normal_channel=args.normal)

    logger.info("The number of training data is: %d", len(TRAIN_DATASET))
    logger.info("The number of test data is: %d", len(TEST_DATASET))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    embs_len = 3
    estimator = PointConvTrans(embs_len).cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        estimator.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(estimator.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            estimator.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    train_steps = 30
    test_steps = 10
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        train_idxs = np.random.choice(TRAIN_DATASET.__len__(), train_steps)
        test_idxs = np.random.choice(TEST_DATASET.__len__(), test_steps)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_idxs)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_idxs)
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize, sampler=train_sampler,
                                                      num_workers=args.num_workers)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, sampler=test_sampler,
                                                     num_workers=args.num_workers)
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):', global_epoch + 1, epoch + 1, args.epoch)
        mean_correct = []

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # for batch_id, data in enumerate(trainDataLoader, 0):
            points_set = data
            points_set = points_set.data.numpy()
            # 增强数据: 随机缩放和平移点云，随机移除一些点
            jittered_data = provider.random_scale_point_cloud(points_set[:, :, 0:3], scale_low=2.0 / 3,
                                                              scale_high=3 / 2.0)
            jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)
            points_set[:, :, 0:3] = jittered_data
            points_set[:, :1024, :] = provider.random_point_dropout_v2(points_set[:, :1024, :])
            points = torch.Tensor(points_set[:, :1024, :])
            target = torch.Tensor(points_set[:, 1024, :])
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            estimator = estimator.train()
            # pred = classifier(points[:, :3, :], points[:, 3:, :])
            pred = estimator(points[:, :3, :], None)
            loss = F.mse_loss(pred, target)
            print(loss)

            # import open3d as o3d
            vis_point = points[0, :, :].data.cpu().numpy().T
            vis_target = target[:1, :].data.cpu().numpy()
            vis_pred = pred[:1, :].data.cpu().numpy()
            print(vis_target, vis_pred)
            # vis_point_cloud = o3d.PointCloud()
            # vis_point_cloud.points = o3d.Vector3dVector(vis_point)
            # vis_point_cloud.paint_uniform_color([1, 0, 0])
            # vis_target_cloud = o3d.PointCloud()
            # vis_target_cloud.points = o3d.Vector3dVector(vis_target)
            # vis_target_cloud.paint_uniform_color([0, 0, 0])
            # vis_pred_cloud = o3d.PointCloud()
            # vis_pred_cloud.points = o3d.Vector3dVector(vis_pred)
            # vis_pred_cloud.paint_uniform_color([0, 0, 1])
            vis_diff = vis_target - vis_pred
            vis_dist = np.linalg.norm(vis_diff)
            print(vis_dist)
            # o3d.draw_geometries([vis_point_cloud, vis_target_cloud, vis_pred_cloud])

            diff = pred - target
            distance = torch.norm(diff, dim=1)
            correct = torch.sum(distance < 0.05)
            mean_correct.append(correct.item() / float(points.size()[0]))
            # loss.backward()
            # optimizer.step()
            global_step += 1

        train_acc = np.mean(mean_correct)
        print('Train Accuracy: %f' % train_acc)
        logger.info('Train Accuracy: %f' % train_acc)

        acc = trans_test(estimator, testDataLoader)

        if (acc >= best_tst_accuracy) and epoch > 5:
            best_tst_accuracy = acc
            logger.info('Save model...')
            save_checkpoint(
                global_epoch + 1,
                train_acc,
                acc,
                estimator,
                optimizer,
                str(checkpoints_dir),
                args.model_name)
            print('Saving model....')

        print('\r Loss: %f' % loss.data)
        logger.info('Loss: %.2f', loss.data)
        print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'), acc, blue('Best Accuracy'), best_tst_accuracy))
        logger.info('Test Accuracy: %f  *** Best Test Accuracy: %f', acc, best_tst_accuracy)

        global_epoch += 1
    print('Best Accuracy: %f' % best_tst_accuracy)

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
