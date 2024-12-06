import os
import numpy as np
from PIL import Image


import torch
from torchvision import transforms


from models import mobilefacenet
from models.sphereface import sphere20, sphere36, sphere64


def extract_deep_features(model, image, device):
    """
    Extracts deep features for an image using the model, including both the original and flipped versions.

    Args:
        model (torch.nn.Module): The pre-trained deep learning model used for feature extraction.
        image (PIL.Image): The input image to extract features from.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.

    Returns:
        torch.Tensor: Combined feature vector of original and flipped images.
    """

    # Define transforms
    original_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    flipped_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Apply transforms
    original_image_tensor = original_transform(image).unsqueeze(0).to(device)
    flipped_image_tensor = flipped_transform(image).unsqueeze(0).to(device)

    # Extract features
    original_features = model(original_image_tensor)
    flipped_features = model(flipped_image_tensor)

    # Combine and return features
    combined_features = torch.cat([original_features, flipped_features], dim=1).squeeze()
    return combined_features


def k_fold_split(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    fold_size = n // n_folds

    for idx in range(n_folds):
        test = base[idx * fold_size:(idx + 1) * fold_size]
        train = base[:idx * fold_size] + base[(idx + 1) * fold_size:]
        folds.append([train, test])

    return folds


def eval_accuracy(predictions, threshold):
    y_true = []
    y_pred = []

    for _, _, distance, gt in predictions:
        y_true.append(int(gt))
        pred = 1 if float(distance) > threshold else 0
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    return accuracy


def find_best_threshold(predictions, thresholds):
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        accuracy = eval_accuracy(predictions, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def eval(model, model_path=None, device=None):
    """
    Evaluates a face verification model on the LFW dataset using pairs.txt.

    Args:
        model (torch.nn.Module): The model to evaluate.
        model_path (str, optional): Path to pre-trained weights. Defaults to None.
        device (torch.device, optional): Device for computation (CPU/GPU). Defaults to auto-detection.

    Returns:
        float: Mean accuracy from K-Fold validation.
        numpy.ndarray: Predictions with image pairs, similarity scores, and ground truth.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, device=device, weights_only=True)['model'])
    model.to(device).eval()

    root = 'data/val'
    with open('data/val/lfw_ann.txt') as f:
        pair_lines = f.readlines()[1:]

    # Extract features and calculate distances
    predicts = []
    with torch.no_grad():
        for line in pair_lines:
            parts = line.strip().split()

            if len(parts) == 3:
                is_same, path1, path2 = parts[0], parts[1], parts[2]
                img1_path = os.path.join(root, path1)
                img2_path = os.path.join(root, path2)
            else:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            # Load and preprocess images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # Extract deep features
            f1 = extract_deep_features(model, img1, device)
            f2 = extract_deep_features(model, img2, device)

            # Compute similarity
            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([img1_path, img2_path, distance.item(), is_same])

    # Convert predictions to numpy array
    predicts = np.array(predicts)

    # Perform K-Fold validation
    thresholds = np.arange(-1.0, 1.0, 0.005)
    accuracies = []
    best_thresholds = []

    folds = k_fold_split(len(predicts), n_folds=10)
    for train_indices, test_indices in folds:
        best_threshold = find_best_threshold(predicts[train_indices], thresholds)
        best_thresholds.append(best_threshold)

        accuracy = eval_accuracy(predicts[test_indices], best_threshold)
        accuracies.append(accuracy)

    # Calculate and display results
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_threshold = np.mean(best_thresholds)

    print(f'LFW ACC: {mean_accuracy:.4f} | STD: {std_accuracy:.4f} Threshold={mean_threshold:.4f}')
    return mean_accuracy, predicts


if __name__ == '__main__':
    _, result = eval(sphere20(512).to('cuda'), model_path='weights/sphere20_last.pth')
    np.savetxt("result.txt", result, '%s')
