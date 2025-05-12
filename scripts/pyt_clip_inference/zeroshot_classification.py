#!/bin/bash

"""
Code adapted from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_classification.py
Thanks to the authors of CLIP_benchmark
"""

import logging
from contextlib import suppress
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    
    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str or dict
        templates to use.
    
    Returns
    -------
    torch.Tensor of shape (N,C) where N is the number
    of templates and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Computing zero-shot classifier"):
            if isinstance(templates, dict):
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif isinstance(templates, list):
                # generic prompts specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor of shape (N, C)
    target: torch.Tensor of shape (N,)
    topk: tuple of int, e.g. (1,5)

    Returns
    -------
    list of top-k accuracies
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True, warmup=False):
    """
    Run zero-shot classification over the dataloader while measuring processing time.

    Parameters
    ----------
    model: torch.nn.Module
        CLIP-like model with `encode_image`
    classifier: torch.Tensor
        obtained from `zero_shot_classifier`
    dataloader: torch.utils.data.DataLoader
    device: cpu or cuda
    amp: bool, whether to use automatic mixed precision
    warmup: bool, if True, run only one iteration (batch) and print latency stats

    Returns
    -------
    If warmup is False, returns a tuple:
       (pred, true, dataloader_len, total_processing_time, avg_latency)
    If warmup is True, prints the stats and returns immediately.
    """

    if not warmup:
        run_classification(model, classifier, dataloader, device, amp, warmup=True)
        print('Warmup is done')
    else:
        print('Warmup is starting')

    dataloader = dataloader_with_indices(dataloader)

    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    total_processing_time = 0.0
    dataloader_len = 0

    with torch.no_grad():
        for images, target, _ in tqdm(dataloader, desc="Running classification") if not warmup else dataloader:
            # dataloader_len += 1
            # t0 = time.time()
            images = images.to(device)
            target = target.to(device)
            with autocast():
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            # batch_time = time.time() - t0
            # total_processing_time += batch_time
            # Print the time for the current iteration
            # print(f"Iteration {dataloader_len}: time = {batch_time:.4f}s")

            true.append(target.cpu())
            pred.append(logits.float().cpu())

            if warmup and dataloader_len >= 2: return


    avg_latency = total_processing_time / dataloader_len if dataloader_len > 0 else 0.0
    return torch.cat(pred), torch.cat(true), dataloader_len, total_processing_time, avg_latency

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def average_precision_per_class(scores, targets):
    """
    Compute average precision for each class (used for multi-label classification).

    Parameters
    ----------
    scores: torch.Tensor of shape (N, C)
    targets: torch.Tensor of shape (N, C)

    Returns
    -------
    torch.Tensor of shape (C,) with average precision for each class.
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    for k in range(scores.size(1)):
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        precision = tp.div(rg)
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True,
             verbose=False, save_clf=None, load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics while also reporting
    dataloader length, total processing time, and average latency.

    Parameters
    ----------
    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    dataloader: torch.utils.data.DataLoader
    tokenizer: text tokenizer
    classnames: list of str
    templates: list of str or dict
    device: cpu or cuda
    amp: bool, whether to use automatic mixed precision
    verbose: bool, whether to print detailed reports
    save_clf: str or None, path to save classifier
    load_clfs: list of paths for loading classifier(s)
    warmup: bool, if True, run only one batch for warmup

    Returns
    -------
    In full evaluation mode (warmup == False): dict of classification metrics including latency stats.
    In warmup mode: runs one batch, prints timing stats, and returns immediately.
    """

    # Build or load the classifier
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)

    if save_clf is not None:
        torch.save(classifier, save_clf)

    # Run the classification evaluation (or warmup)
    result = run_classification(model, classifier, dataloader, device, amp=amp)

    # Unpack the results from full evaluation mode
    logits, target, dataloader_len, total_processing_time, avg_latency = result

    is_multilabel = (len(target.shape) == 2)
    metrics = {}
    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        metrics["mean_average_precision"] = ap_per_class.mean().item()
    else:
        pred = logits.argmax(axis=1)
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan")
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        metrics["acc1"] = acc1
        metrics["acc5"] = acc5
        metrics["mean_per_class_recall"] = mean_per_class_recall

    # Add latency statistics to the returned metrics
    metrics["dataloader_len"] = dataloader_len
    metrics["total_processing_time"] = total_processing_time
    metrics["avg_latency"] = avg_latency

    print(f"dataloader_len={dataloader_len}, total_processing_time={total_processing_time:.4f}s, avg_latency={avg_latency:.4f}s")
    return metrics

