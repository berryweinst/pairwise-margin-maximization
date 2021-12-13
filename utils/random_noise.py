import os
import numpy as np


def get_mean_std(dataset='cifar10'):
    if dataset == 'cifar10' or dataset == 'cifar100':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(dataset))
    return mean, std


def label_noise(dataset, noise_rate=0.1):
    if noise_rate > 1:
        raise ValueError("Noise rate {} not supported. Using clean dataset.".format(args.noise_rate))

    np.random.seed(1234)

    train_labels = np.asarray(dataset.targets)
    train_labels_old = np.copy(train_labels)

    # setup
    print("Randomizing {:.1f} percent of labels ".format(noise_rate * 100))

    # randomize labels with given noise rate
    n_train = len(train_labels)
    n_rand = int(noise_rate * n_train)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
    train_labels[randomize_indices] = np.random.choice(range(dataset.num_classes), size=n_rand, replace=True)

    wrong_indices = np.where(train_labels != train_labels_old)[0]
    wrong_labels = train_labels[wrong_indices]


    # # save noisy labels and indices to file
    # if hasattr(args, 'save_dir'):
    #     out_fname = os.path.join(args.save_dir, 'noisy_idx_labels.npy')
    #     np.save(out_fname, np.stack([wrong_indices, wrong_labels], axis=1))
    #     print("Noisy labels saved to {}".format(out_fname))
    # else:
    #     print("Noise info is not saved.")

    # print info
    print("Size of dataset: {}.".format(len(dataset.data)))
    print("Label error rate: {:.2f}.".format(wrong_indices.shape[0] * 1. / len(dataset.data)))

    # apply the change to original dataset
    dataset.targets = train_labels.tolist()

    return wrong_indices


def image_noise(dataset, noise_rate=0.2):
    # if noise_rate > 1:
    #     raise ValueError("Noise rate {} not supported. Using clean dataset.".format(args.noise_rate))

    np.random.seed(1234)

    mean, std = get_mean_std()
    train_images = dataset.data

    # if args.noise_info is None:
    # setup
    # print("Randomizing {:.1f} percent of images with `{}` scheme.".format(args.noise_rate * 100, args.noise_type))

    # randomize labels with given noise rate
    n_train = len(train_images)
    n_rand = int(noise_rate * n_train)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)

    # if args.noise_type == 'Gaussian':
    # generate examples by Gaussian distribution with the same mean and std as original dataset
    randomized_images = np.random.randn(*train_images[randomize_indices].shape)
    randomized_images *= np.asarray(std)
    randomized_images += np.asarray(mean)

    # convert to image format
    randomized_images *= 255
    randomized_images = np.clip(randomized_images, a_min=0, a_max=255)
    randomized_images = randomized_images.astype(np.uint8)

    # elif args.noise_type == 'random_pixels':
    #     randomized_images = train_images[randomize_indices].copy()
    #     n, h, w, c = randomized_images.shape
    #     for i in range(n):
    #         shuffle_matrix = np.arange(h * w, dtype=np.int)
    #         np.random.shuffle(shuffle_matrix)
    #         randomized_images[i] = randomized_images[i].reshape(h * w, c)[shuffle_matrix].reshape(h, w, c)
    #
    # elif args.noise_type == 'shuffled_pixels':
    #     randomized_images = train_images[randomize_indices].copy()
    #     n, h, w, c = randomized_images.shape
    #     randomized_images = randomized_images.transpose(1, 2, 0, 3).reshape(h * w, n * c)
    #     shuffle_matrix = np.arange(h * w, dtype=np.int)
    #     np.random.shuffle(shuffle_matrix)
    #     randomized_images = randomized_images[shuffle_matrix]
    #     randomized_images = randomized_images.reshape(h, w, n, c).transpose(2, 0, 1, 3)
    #
    # else:
    #     raise ValueError("Noise type {} is not supported yet.".format(args.noise_type))

    # else:
    #     print("Using noise info specified in {}".format(args.noise_info))
    #     if not os.path.isdir(args.noise_info):
    #         args.noise_info = os.path.dirname(args.noise_info)
    #     in_fname = os.path.join(args.noise_info, 'noisy_idx.npy')
    #     randomize_indices = np.load(in_fname)
    #     in_fname = os.path.join(args.noise_info, 'noisy_images.npy')
    #     randomized_images = np.load(in_fname)

    train_images[randomize_indices] = randomized_images

    # save noisy labels and indices to file
    # if hasattr(args, 'save_dir'):
    #     out_fname = os.path.join(args.save_dir, 'noisy_idx.npy')
    #     np.save(out_fname, randomize_indices)
    #     out_fname = os.path.join(args.save_dir, 'noisy_images.npy')
    #     np.save(out_fname, randomized_images)
    #     print("Noise info is saved to {}".format(args.save_dir))
    # else:
    #     print("Noise info is not saved.")

    # print info
    print("Size of dataset: {}.".format(len(dataset.data)))

    return randomize_indices