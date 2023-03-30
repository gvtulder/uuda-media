# Copyright (C) 2023 Gijs van Tulder
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import collections
import time
import json
import os.path
import sys
import numpy as np
import torch
import torch.utils.data
import argparse
from revgrad import revgrad
import models
import data
import similarity

torch.set_num_threads(1)
np.set_printoptions(suppress=True, linewidth=160)

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-idx', type=int)
parser.add_argument('--model', default='SingleDense',
                               choices=list(models.classes.keys()))
parser.add_argument('--epochs', metavar='N', type=int, default=1000)
parser.add_argument('--mb-size', metavar='N', type=int, default=32)
parser.add_argument('--steps-per-epoch', metavar='N', type=int, default=100)
parser.add_argument('--learning-rate-class', metavar='LR', type=float, default=1e-2)
parser.add_argument('--learning-rate-adv', metavar='LR', type=float, default=1e-2)
parser.add_argument('--classification-weight', metavar='F', type=float, default=1.0)
parser.add_argument('--adversarial-weight-A', metavar='F', type=float, default=1.0)
parser.add_argument('--adversarial-weight-B', metavar='F', type=float, default=1.0)
parser.add_argument('--class-balance', metavar='F', type=float, default=0.5)
parser.add_argument('--optimizer', choices=('sgd', 'adam', 'adadelta'), default='sgd')
parser.add_argument('--alternated-training', action='store_true')
parser.add_argument('--revgrad', action='store_true')
parser.add_argument('--delay-adversarial', type=int, default=0)
parser.add_argument('--freeze-classification', type=int, default=None)
parser.add_argument('--two-phase', type=int, help='train domain A first, then B after this epoch')
parser.add_argument('--common-training-samples', action='store_true')
parser.add_argument('--single-training-batch', action='store_true')
parser.add_argument('--shared-initialization', action='store_true')
parser.add_argument('--adversarial-target',
                    choices=('binary', '90', 'max', 'confusion'), default='binary')
parser.add_argument('--discriminator-loss-fn', choices=('bce', 'mse'), default='bce')
parser.add_argument('--device', metavar='DEVICE', type=str, default='cpu')
parser.add_argument('--num-workers', metavar='N', type=int, default=0)
parser.add_argument('--data', default='SyntheticTwo',
                    choices=list(data.classes.keys()))
parser.add_argument('--output-directory', metavar='DIR', type=str, default='results/')
args = parser.parse_args()


if args.experiment_idx:
    if os.path.exists('%s/%04d.npz' % (args.output_directory, args.experiment_idx)):
        print('Already run.')
        sys.exit()


if args.two_phase is not None:
    assert args.adversarial_weight_A == args.adversarial_weight_B
    assert args.revgrad
    assert not args.alternated_training

elif args.revgrad:
    assert args.learning_rate_class == args.learning_rate_adv
    assert args.adversarial_weight_A == args.adversarial_weight_B
    assert not args.alternated_training


dtype = torch.float
device = torch.device(args.device)

# build data sources
print('Loading data')
if args.data in data.classes:
    data_class = data.classes[args.data]
    data_sources = {}
    if 'BRATS' in args.data:
        subsets = ('train_A', 'train_B', 'validation_A', 'validation_B', 'test_A', 'test_B')
        phases = ('train', 'validation', 'test')
    else:
        subsets = ('train_A', 'train_B', 'validation_A', 'validation_B')
        phases = ('train', 'validation')
    for subset in subsets:
        data_sources[subset] = data_class(class_balance=args.class_balance, subset=subset)
else:
    raise Exception('Unknown data source "%s"' % args.data)

number_of_classes = data_sources['train_A'].number_of_classes
number_of_groups = data_sources['train_A'].number_of_groups
patch_size = data_sources['train_A'].patch_size


def worker_init_fn(w):
    # workers need different random seeds to produce independent samples
    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    np.random.seed(worker_seed)


# build data loaders
print('Constructing data loaders')
data_loaders = {}
for key, ds in data_sources.items():
    data_loaders[key] = torch.utils.data.DataLoader(ds,
                                                    batch_size=args.mb_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    worker_init_fn=worker_init_fn)

# build model
print('Constructing model')
if args.model in models.classes:
    model = models.classes[args.model](patch_size, device)
else:
    raise Exception('Unknown model "%s"' % args.model)


# move to cuda (again, if necessary)
encoders = model.encoders.to(device)
classifier = model.classifier.to(device)
discriminator = model.discriminator.to(device)

# loss functions
if number_of_classes == 2:
    classifier_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
else:
    classifier_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
adversarial_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
if args.discriminator_loss_fn == 'bce':
    discriminator_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
else:
    assert args.discriminator_loss_fn == 'mse'
    discriminator_loss_fn = torch.nn.MSELoss(reduction='sum')
representation_difference_fn = torch.nn.MSELoss()

# parameters
all_classifier_parameters = model.classifier_parameters()
all_discriminator_parameters = model.discriminator_parameters()
if args.shared_initialization:
    model.encoders.share_parameters()

# optimizers
print('Constructing optimizers')
if args.two_phase is None:
    # concurrent optimization
    if args.optimizer == 'adam':
        classifier_optimizer = torch.optim.Adam(all_classifier_parameters,
                                                lr=args.learning_rate_class)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                                   lr=args.learning_rate_adv)
    elif args.optimizer == 'adadelta':
        classifier_optimizer = torch.optim.Adadelta(all_classifier_parameters)
        discriminator_optimizer = torch.optim.Adadelta(discriminator.parameters())
    elif args.optimizer == 'sgd':
        classifier_optimizer = torch.optim.SGD(all_classifier_parameters,
                                               lr=args.learning_rate_class)
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(),
                                                  lr=args.learning_rate_adv)
    else:
        raise Exception('Unknown optimizer %s' % args.optimizer)
else:
    # two-phase optimization: first encoder A + classifier,
    # then encoder B + discriminator
    if args.optimizer == 'adam':
        phaseA_optimizer = torch.optim.Adam(list(model.encoderA.parameters()) +
                                            list(model.classifier.parameters()),
                                            lr=args.learning_rate_class)
        phaseB_optimizer = torch.optim.Adam(list(model.encoderB.parameters()) +
                                            list(model.discriminator.parameters()),
                                            lr=args.learning_rate_adv)
    elif args.optimizer == 'sgd':
        phaseA_optimizer = torch.optim.SGD(list(model.encoderA.parameters()) +
                                           list(model.classifier.parameters()),
                                           lr=args.learning_rate_class)
        phaseB_optimizer = torch.optim.SGD(list(model.encoderB.parameters()) +
                                           list(model.discriminator.parameters()),
                                           lr=args.learning_rate_adv)
    else:
        raise Exception('Unknown optimizer %s' % args.optimizer)


def accuracy(a, b):
    a = a.detach().cpu().numpy().astype(int).flatten()
    b = b.detach().cpu().numpy().astype(int).flatten()
    acc = np.mean((a == b).astype(float))
    return acc


def get_one_hot(targets, num_classes):
    res = np.eye(num_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [num_classes])


def confusion_matrix(prediction, truth, prop=False):
    prediction = prediction.detach().cpu().numpy().astype(int).flatten()
    truth = truth.detach().cpu().numpy().astype(int).flatten()
    scores = np.zeros((np.max(prediction) + 1, np.max(truth) + 1), dtype=int)
    scores = np.zeros((number_of_classes, number_of_groups), dtype=int)
    for p, t in zip(prediction, truth):
        scores[p, t] += 1
    if prop:
        scores = scores.astype(float) / len(prediction)
    return scores


losses_history = collections.defaultdict(list)


# optional:
# train with a single training batch
if args.single_training_batch:
    assert False

# optional:
# use the same training samples for both domains
if args.common_training_samples:
    assert False


# run the experiment
print('Start training')
for epoch in range(args.epochs):
    start_time = time.time()
    losses = collections.defaultdict(float)
    steps = collections.defaultdict(int)

    for phase in phases:
        all_features_A_in_A, all_features_B_in_A = [], []
        all_features_A_in_B, all_features_B_in_B = [], []

        for batch_A, batch_B in zip(data_loaders['%s_A' % phase],
                                    data_loaders['%s_B' % phase]):
            # unpack batches
            x_A, labels_A, groups_A = batch_A
            x_B, labels_B, groups_B = batch_B

            steps[phase] += 1

            if phase == 'train':
                model.train()
            else:
                model.eval()

            # preprocess and transfer data
            x_A = x_A.to(dtype=dtype, device=device)
            x_B = x_B.to(dtype=dtype, device=device)
            if number_of_classes == 2:
                labels_A = labels_A[:, None].to(dtype=dtype, device=device)
                labels_B = labels_B[:, None].to(dtype=dtype, device=device)
            else:
                labels_oh_A = torch.tensor(get_one_hot(labels_A, num_classes=number_of_classes),
                                           dtype=dtype, device=device)
                labels_oh_B = torch.tensor(get_one_hot(labels_B, num_classes=number_of_classes),
                                           dtype=dtype, device=device)
                labels_A = labels_A.to(dtype=torch.long, device=device)
                labels_B = labels_B.to(dtype=torch.long, device=device)
            domains_A = torch.zeros([labels_A.shape[0], 1], dtype=dtype, device=device)
            domains_B = torch.ones([labels_B.shape[0], 1], dtype=dtype, device=device)

            # forward pass: compute classification of inputs A and B, discriminator score
            enc_A, enc_B = model.encoders(x_A, x_B)
            y_pred_A = model.classifier(enc_A)
            y_pred_B = model.classifier(enc_B)

            if args.revgrad:
                discr_A = model.discriminator(revgrad(enc_A))
                discr_B = model.discriminator(revgrad(enc_B))
            else:
                discr_A = model.discriminator(enc_A)
                discr_B = model.discriminator(enc_B)

            # compute classification loss
            classification_loss_A = classifier_loss_fn(y_pred_A, labels_A)
            classification_loss_B = classifier_loss_fn(y_pred_B, labels_B)
            losses['classification_loss_A/%s' % phase] += classification_loss_A.item()
            losses['classification_loss_B/%s' % phase] += classification_loss_B.item()

            if number_of_classes == 2:
                losses['classification_acc_A/%s' % phase] += accuracy(y_pred_A > 0, labels_A > 0.5)
                losses['classification_acc_B/%s' % phase] += accuracy(y_pred_B > 0, labels_B > 0.5)

                losses['confmat_A/%s' % phase] += confusion_matrix((y_pred_A > 0), groups_A, prop=True)
                losses['confmat_B/%s' % phase] += confusion_matrix((y_pred_B > 0), groups_B, prop=True)
            else:
                y_pred_label_A = torch.argmax(y_pred_A, dim=1)
                y_pred_label_B = torch.argmax(y_pred_B, dim=1)
                losses['classification_acc_A/%s' % phase] += accuracy(y_pred_label_A, labels_A)
                losses['classification_acc_B/%s' % phase] += accuracy(y_pred_label_B, labels_B)

                losses['confmat_A/%s' % phase] += confusion_matrix(y_pred_label_A, groups_A, prop=True)
                losses['confmat_B/%s' % phase] += confusion_matrix(y_pred_label_B, groups_B, prop=True)

            # compute discriminator loss
            if args.adversarial_target == '90':
                discriminator_loss_A = discriminator_loss_fn(discr_A, domains_A + 0.1)
                discriminator_loss_B = discriminator_loss_fn(discr_B, domains_B - 0.1)
            else:
                discriminator_loss_A = discriminator_loss_fn(discr_A, domains_A)
                discriminator_loss_B = discriminator_loss_fn(discr_B, domains_B)
            losses['discriminator_loss_A/%s' % phase] += discriminator_loss_A.item()
            losses['discriminator_loss_B/%s' % phase] += discriminator_loss_B.item()

            losses['discriminator_acc_A/%s' % phase] += accuracy(discr_A > 0, domains_A > 0.5)
            losses['discriminator_acc_B/%s' % phase] += accuracy(discr_B > 0, domains_B > 0.5)

            # compute difference in representations
            with torch.no_grad():
                # encode samples from A
                dummy_enc_A, dummy_enc_B = encoders(x_A, x_A)
                losses['representation_difference_A/%s' % phase] += representation_difference_fn(dummy_enc_A, dummy_enc_B).item()
                # keep computed features for CKA
                all_features_A_in_A.append(dummy_enc_A.flatten(1).detach().cpu().numpy())
                all_features_A_in_B.append(dummy_enc_B.flatten(1).detach().cpu().numpy())

                # encode samples from B
                dummy_enc_A, dummy_enc_B = encoders(x_B, x_B)
                losses['representation_difference_B/%s' % phase] += representation_difference_fn(dummy_enc_A, dummy_enc_B).item()
                # keep computed features for CKA
                all_features_B_in_A.append(dummy_enc_A.flatten(1).detach().cpu().numpy())
                all_features_B_in_B.append(dummy_enc_B.flatten(1).detach().cpu().numpy())

            # two options to train with adversarial model:
            # reverse gradient or with a separate adversarial loss
            if args.revgrad:
                # reverse gradients
                combined_loss_components = []
                if args.two_phase is None:
                    # train encoder A and B at the same time
                    if args.freeze_classification is None or epoch < args.freeze_classification:
                        # classification training objective
                        combined_loss_components.append(args.classification_weight * classification_loss_A)
                    if args.delay_adversarial <= epoch:
                        # loss for adversarial training
                        combined_loss_components.append(args.adversarial_weight_A * discriminator_loss_A +
                                                        args.adversarial_weight_B * discriminator_loss_B)
                    combined_loss = combined_loss_components[0]
                    # note: this adds the classification loss twice
                    for tc in combined_loss_components:
                        combined_loss += tc
                    losses['combined_loss/%s' % phase] += combined_loss.item()

                    assert not args.alternated_training

                    if phase == 'train':
                        # compute gradient updates
                        classifier_optimizer.zero_grad()
                        discriminator_optimizer.zero_grad()
                        combined_loss.backward()
                        discriminator_optimizer.step()
                        classifier_optimizer.step()
                else:
                    # train encoder A and classifier first,then encoder B and discriminator
                    assert args.freeze_classification is None
                    assert args.delay_adversarial == 0

                    if epoch < args.two_phase:
                        # first phase: train the classifier and encoder A
                        combined_loss = classification_loss_A
                        if phase == 'train':
                            phaseA_optimizer.zero_grad()
                            classification_loss_A.backward()
                            phaseA_optimizer.step()
                    else:
                        # second phase: train the discriminator and encoder B
                        combined_loss = discriminator_loss_A + discriminator_loss_B
                        if phase == 'train':
                            phaseB_optimizer.zero_grad()
                            combined_loss.backward()
                            phaseB_optimizer.step()

                    losses['combined_loss/%s' % phase] += combined_loss.item()

            else:
                # compute adversarial loss for classifier training
                if args.adversarial_target == '90':
                    adversarial_loss_A = adversarial_loss_fn(discr_A, torch.ones_like(domains_A) - 0.1)
                    adversarial_loss_B = adversarial_loss_fn(discr_B, torch.zeros_like(domains_B) + 0.1)
                elif args.adversarial_target == 'confusion':
                    # domain confusion objective
                    adversarial_loss_A = adversarial_loss_fn(discr_A, 0.5 * torch.ones_like(domains_A))
                    adversarial_loss_B = adversarial_loss_fn(discr_B, 0.5 * torch.ones_like(domains_B))
                elif args.adversarial_target == 'max':
                    # maximise instead of minimise
                    adversarial_loss_A = -adversarial_loss_fn(discr_A, domains_A)
                    adversarial_loss_B = -adversarial_loss_fn(discr_B, domains_B)
                else:
                    adversarial_loss_A = adversarial_loss_fn(discr_A, torch.ones_like(domains_A))
                    adversarial_loss_B = adversarial_loss_fn(discr_B, torch.zeros_like(domains_B))

                combined_loss_components = []
                if args.freeze_classification is None or epoch < args.freeze_classification:
                    # classification training objective
                    combined_loss_components.append(args.classification_weight * classification_loss_A)
                if args.delay_adversarial <= epoch:
                    # loss for adversarial training
                    combined_loss_components.append(args.adversarial_weight_A * adversarial_loss_A +
                                                    args.adversarial_weight_B * adversarial_loss_B)
                combined_loss = combined_loss_components[0]
                for tc in combined_loss_components:
                    combined_loss += tc
                losses['combined_loss/%s' % phase] += combined_loss.item()

                if phase == 'train':
                    if not args.alternated_training or t % 2 == 0:
                        # compute gradient updates
                        classifier_optimizer.zero_grad()
                        combined_loss.backward()
                        classifier_optimizer.step()

                    if (args.delay_adversarial <= epoch) and (not args.alternated_training or t % 2 == 1):
                        # train discriminator
                        discriminator_optimizer.zero_grad()
                        (discriminator_loss_A + discriminator_loss_B).backward()
                        discriminator_optimizer.step()

        for k in losses.keys():
            if phase in k:
                losses[k] = losses[k] / steps[phase]
                losses_history[k].append(losses[k])

        # compute linear CKA
        all_features_A_in_A = np.concatenate(all_features_A_in_A)
        all_features_A_in_B = np.concatenate(all_features_A_in_B)
        all_features_B_in_A = np.concatenate(all_features_B_in_A)
        all_features_B_in_B = np.concatenate(all_features_B_in_B)
        losses['linear_cka_A/%s' % phase] = similarity.feature_space_linear_cka(all_features_A_in_A, all_features_A_in_B)
        losses['linear_cka_B/%s' % phase] = similarity.feature_space_linear_cka(all_features_B_in_A, all_features_B_in_B)
        losses_history['linear_cka_A/%s' % phase].append(losses['linear_cka_A/%s' % phase])
        losses_history['linear_cka_B/%s' % phase].append(losses['linear_cka_B/%s' % phase])
        del all_features_A_in_A, all_features_A_in_B
        del all_features_B_in_A, all_features_B_in_B

    for phase in phases:
        print('Epoch %4d [%0.2fs]: [%s] combined loss=%6.2f  classif loss=[A: %6.2f, B: %6.2f]  discrim loss=[A: %6.2f, B: %6.2f]  classif acc=[A: %6.2f, B: %6.2f]  discrim acc=[A: %6.2f, B: %6.2f]  representation_diff=[A: %6.2f, B: %6.2f]  linear_cka=[A: %6.2f, B: %6.2f]'
              % (epoch, time.time() - start_time,
                 phase,
                 losses['combined_loss/%s' % phase],
                 losses['classification_loss_A/%s' % phase],
                 losses['classification_loss_B/%s' % phase],
                 losses['discriminator_loss_A/%s' % phase],
                 losses['discriminator_loss_B/%s' % phase],
                 losses['classification_acc_A/%s' % phase],
                 losses['classification_acc_B/%s' % phase],
                 losses['discriminator_acc_A/%s' % phase],
                 losses['discriminator_acc_B/%s' % phase],
                 losses['representation_difference_A/%s' % phase],
                 losses['representation_difference_B/%s' % phase],
                 losses['linear_cka_A/%s' % phase],
                 losses['linear_cka_B/%s' % phase]))
        print((losses['confmat_A/%s' % phase] * 1000).astype(int))
        print((losses['confmat_B/%s' % phase] * 1000).astype(int))

for k in losses_history.keys():
    losses_history[k] = np.array(losses_history[k])

if args.experiment_idx:
    with open('%s/%04d-params.json' % (args.output_directory, args.experiment_idx), 'w') as f:
        json.dump(vars(args), f)
    np.savez_compressed('%s/%04d.npz' % (args.output_directory, args.experiment_idx), **losses_history)
