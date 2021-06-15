
import os
import argparse
import csv

import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from resnet import resnet2b, resnet4b
from eval import normalize
import numpy as np
import torch.nn.functional as F


def attack_pgd(model, X, y, epsilon, alpha=0.5/255, attack_iters=50, restarts=5, target=None, upper_limit=None, lower_limit=None):
    # batch = 1, target pgd attack
    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    model = model.cuda()
    epsilon = epsilon.cuda()
    X = X.cuda()
    max_loss = torch.zeros(y.shape[0]).cuda()-100.
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        # alpha = np.random.uniform(alpha-0.2/255, alpha+0.2/255)
        # delta = torch.zeros_like(X).float().cuda()
        delta = (upper_limit - lower_limit) * torch.rand(size=X.shape, device='cuda') + lower_limit
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            index = slice(None, None, None)
            if target is None:
                loss = F.cross_entropy(output, y)
            else:
                # target logit guided
                loss = -F.cross_entropy(output, target)

            loss.backward()

            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            d = torch.max(torch.min(d + alpha * torch.sign(g), epsilon), -epsilon)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
            if target is None:
                all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
            else:
                all_loss = -F.cross_entropy(model(normalize(X + delta)), target, reduction='none')
        if all_loss[0] > max_loss[0]:
            max_delta = delta.clone().detach()
        max_loss = torch.max(max_loss, all_loss)

    assert delta.abs().max()<=epsilon

    output = model(normalize(torch.max(torch.min(X+max_delta, upper_limit), lower_limit)))
    if output.argmax(1) == target:
        return True  # attack success
    else:
        return False


def load_data(data_dir: str = "./tmp", num_imgs: int = 25, random: bool = False) -> tuple:

    """
    Loads the cifar10 data.

    Args:
        data_dir:
            The directory to store the full CIFAR10 dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs)

    return next(iter(loader_test))


# noinspection PyShadowingNames
def create_input_bounds(img: torch.Tensor, eps: float,
                        mean: tuple = (0.4914, 0.4822, 0.4465),
                        std: tuple = (0.2471, 0.2435, 0.2616)) -> torch.Tensor:

    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
        mean:
            The channel-wise means.
        std:
            The channel-wise standard deviation.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """

    mean = torch.Tensor(mean).view(-1, 1, 1)
    std = torch.Tensor(std).view(-1, 1, 1)

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = (torch.clip((img - eps), 0, 1) - mean) / std
    bounds[..., 1] = (torch.clip((img + eps), 0, 1) - mean) / std
    # print(bounds[..., 0].abs().sum(), bounds[..., 1].abs().sum())

    return bounds.view(-1, 2)


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, runnerup: int, spec_path: str, total_output_class: int = 10):

    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """

    with open(spec_path, "w") as f:

        f.write(f"; CIFAR10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        # orignal separate version:
        # for i in range(total_output_class):
        #     if i != label:
        #         f.write(f"(assert (>= Y_{label} Y_{i}))\n")
        # f.write("\n")

        # disjunction version:
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")

def create_csv():
    name = ["model_name", "property_name", "timeout"]
    instance_list = []

    # 48 properties for resnet2b
    model_name = "resnet_2b"
    assert os.path.exists(f"../onnx/{model_name}.onnx")
    assert os.path.exists("../vnnlib_properties_pgd_filtered/")
    for i in range(48):
        instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_{i}_eps_0.008.vnnlib", "300"])

    # 24 properties for resnet2b
    model_name = "resnet_4b"
    assert os.path.exists(f"../onnx/{model_name}.onnx")
    for i in range(24):
        instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet4b_pgd_filtered/prop_{i}_eps_0.004.vnnlib", "300"])

    with open('../instance.csv', 'w') as f:
        write = csv.writer(f)
        # write.writerow(fields)
        write.writerows(instance_list)


def create_vnnlib(args):
    num_imgs = args.num_images
    random = args.random
    print(f"===== model: {args.model} epsilons: {args.epsilons} total images: {args.num_images} =====")
    print("randomness:", args.random, "seed:", args.seed)
    epsilons = [eval(eps) for eps in args.epsilons.split(" ")]

    result_dir = "../vnnlib_properties_pgd_filtered/"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    result_dir = os.path.join("../vnnlib_properties_pgd_filtered/", args.model+"_pgd_filtered/")
    model_path = os.path.join(args.model, "model_best.pth")

    print("loading model {} and properties saved in {}".format(model_path, result_dir))

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    model = eval(args.model)()
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model = model.cuda()

    if args.random and args.seed is not None:
        # we use random seed 0 for deterministic testing
        torch.random.manual_seed(args.seed)

    images, labels = load_data(num_imgs=10000, random=random)

    for eps in epsilons:
        acc, pgd_acc = 0, 0
        cnt = 0
        for i in range(images.shape[0]):
            if cnt>=num_imgs:
                break

            image, label = images[i], labels[i]
            # adding pgd filter targeting for runner up label
            output = model(normalize(image.unsqueeze(0).cuda()))
            if output.max(1)[1] != label: 
                print("incorrect image {}".format(i))
                continue
            acc += 1
            # continue
            output[0, label] = -np.inf

            #########
            # runnerup label targeted pgd
            # runnerup = output.max(1)[1].item()
            # print("image {}/{} label {} runnerup {}".format(cnt, i, label, runnerup))

            # pgd_success = attack_pgd(model, X=image.unsqueeze(0), y=torch.tensor([label], device="cuda"),
            #                      epsilon=torch.tensor(eps, device="cuda"), upper_limit=torch.tensor(1., device="cuda"), 
            #                      lower_limit=torch.tensor(0., device="cuda"),
            #                      target=torch.tensor([runnerup], device="cuda"))
            # if pgd_success:
            #     print("pgd succeed {}".format(i))
            #     continue

            #########
            # All label targeted pgd
            pgd_success = False
            for runnerup in range(10):
                if runnerup == label:
                    continue
                print("image {}/{} label {} target label {}".format(cnt, i, label, runnerup))

                pgd_success = attack_pgd(model, X=image.unsqueeze(0), y=torch.tensor([label], device="cuda"),
                                     epsilon=torch.tensor(eps, device="cuda"), upper_limit=torch.tensor(1., device="cuda"), 
                                     lower_limit=torch.tensor(0., device="cuda"),
                                     target=torch.tensor([runnerup], device="cuda"))
                if pgd_success:
                    break
                    
            if pgd_success:
                print("pgd succeed image {}, label {}, against label {}".format(i, label, runnerup))
                continue

            pgd_acc += 1
            # continue

            input_bounds = create_input_bounds(image, eps)

            spec_path = os.path.join(result_dir, f"prop_{cnt}_eps_{eps:.3f}.vnnlib")

            save_vnnlib(input_bounds, label, runnerup, spec_path)
            cnt += 1

    print("acc:", acc, "pgd_acc:", pgd_acc, "out of", i, "samples")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet2b", choices=["resnet2b", "resnet4b"])
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--random', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epsilons', type=str, default="2/255")
    args = parser.parse_args()

    # Example: $python generate_properties_pgd.py --num_images 100 --random True --epsilons '2/255' --seed 0

    args.model = "resnet2b"
    args.epsilons = "2/255"
    args.num_images = 48
    create_vnnlib(args)

    args.model = "resnet4b"
    args.epsilons = "1/255"
    args.num_images = 24
    create_vnnlib(args)

    create_csv()


