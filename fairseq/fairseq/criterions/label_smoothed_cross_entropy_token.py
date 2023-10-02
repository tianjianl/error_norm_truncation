# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
import gc

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from collections import defaultdict

@dataclass
class LabelSmoothedCrossEntropyTokenCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

def compute_el2n(probs, target, log_probs=False, ignore_indices=None):
    
    """
    Computes the 2 norm of the error vector probs - target_one_hot, ignoring indices where target = 1.

    Args:
    - probs: torch.Tensor, shape of (batch size, length, vocab size)
    - target: torch.Tensor, shape of (batch size, length)
    - log_probs: boolean, indicating whether the input probs tensor is log probs or raw probs
    - ignore_indices: list of indices in the batch to ignore, e.g. [0, 1, 3, 5] ignores the first, second, fourth, and sixth data points in the batch

    Returns:
    - el2n: torch.Tensor, shape of (batch size, length)
    """

    if log_probs:
        probs = torch.exp(probs)

    vocab_size = probs.size(-1)
    target_one_hot = F.one_hot(target, vocab_size)
    
    # Masking out the <pad> tokens and setting their probabilities to zero
    mask = target == 1
    target_one_hot[mask, :] = 0
    probs[mask, :] = 0

    # Masking out the indices to ignore
    if ignore_indices != None and len(ignore_indices) > 0:
        mask = torch.zeros_like(target_one_hot, dtype=torch.bool)
        mask[ignore_indices, :] = 1
        target_one_hot[mask] = 0
        probs[mask] = 0
    
    el2n = torch.linalg.norm(probs - target_one_hot, dim=-1)
    return el2n


def label_smoothed_nll_loss(lprobs, batched_lprobs, target, batched_target, epsilon, ignore_index=None, reduce=True, threshold=None, threshold_type='el2n', ignore_batch_indices=None, prune_fraction=0):
    
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    
    # Mask out loss where predicted probability is below threshold
    if threshold != None or prune_fraction != 0:
        if threshold != None and prune_fraction != 0:
            raise Exception("Sorry, one one of threshold pruning and fraction pruning can be used")
        
        if threshold_type == 'el2n':
            
            el2n = compute_el2n(batched_lprobs, batched_target, log_probs=True, ignore_indices=ignore_batch_indices)
            # el2n is of shape (batch size, length)
            
            if threshold != None:
                mask = el2n > threshold
                mask = mask.view(-1).unsqueeze(dim=-1)
            elif prune_fraction != 0:
                batch_size = batched_lprobs.shape[0]
                el2n = el2n.view(-1)
                sorted_el2n = torch.sort(el2n, descending=True).values
                threshold = sorted_el2n[int(prune_fraction * len(sorted_el2n))]
                el2n = el2n.view(batch_size, -1)
                mask = el2n > threshold
                mask = mask.view(-1).unsqueeze(dim=-1)   
        elif threshold_type == 'prob':
            if ignore_batch_indices != None:
                prob = torch.exp(batched_lprobs)
                prob = prob.gather(dim=-1, index=target)
                # set the probability of the ignore indices to be 1 to prevent it from being masked
                prob[ignore_batch_indices, :] = 1.0
                mask = prob < threshold
                mask = mask.view(-1).unsqueeze(dim=-1)
            else:
                prob = torch.exp(lprobs)
                prob = prob.gather(dim=-1, index=target)
                mask = prob < threshold

        nll_loss.masked_fill_(mask, 0.0)
        smooth_loss.masked_fill_(mask, 0.0)
    
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    return loss, nll_loss

@register_criterion(
        "label_smoothed_cross_entropy_token", dataclass=LabelSmoothedCrossEntropyTokenCriterionConfig
)
class LabelSmoothedCrossEntropyTokenCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        

    def forward(self, model, sample, reduce=True, ignore_low_probs=None, metric='el2n', length_prune=0, mask_input=False, prune_languages='all', prune_fraction=0):

        """
        Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output = model(**sample["net_input"])
    
        # this has pad tokens in the end 
        # print(sample["target"])  
        # this has pad tokens at the beginning
        # print(f'original sample = {sample["net_input"]["prev_output_tokens"]}') 

        # sample["net_input"] keys = "src_tokens" "src_lengths" "prev_output_tokens"   
        # get batch index for pruned langauges 
        
        indices = None
        if prune_languages != 'all':
            lang_toks = sample["net_input"]["src_tokens"][:, 0] 
            #print(f"lang_toks = {lang_toks}")
            indices = [idx for idx, lang_tok in enumerate(lang_toks) if lang_tok not in prune_languages]
     
        
        if mask_input != None and ignore_low_probs != None: #masking the input and the output
            
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            el2n = compute_el2n(lprobs, sample['target'], log_probs=True, ignore_batch_indices=indices)
            mask = el2n > ignore_low_probs
            
            # print(f"mask.shape = {mask.shape}")  
            # (f"mask = {mask}")

            # shift the mask to the left by 1
            mask = torch.cat((torch.zeros((mask.shape[0], 1), dtype=torch.bool, device='cuda:0'), mask[:, :-1]), dim=1)
            # print(f"shifted mask = {mask}")
            
            # set the input to self.padding_idx if the predicted probability is below threshold
            sample['net_input']['prev_output_tokens'].masked_fill_(mask, self.padding_idx)
            # print(f'masked sample = {sample["net_input"]["prev_output_tokens"]}')
            
            # set the target index to self.padding_idx as well
            if mask_input == 'both':
                sample['target'].masked_fill_(mask, self.padding_idx)

            # recompute the net_output again and use standard training
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, 
                                               threshold=ignore_low_probs, 
                                               metric=metric, 
                                               length_prune=length_prune, 
                                               ignore_batch_indices=indices,
                                               prune_fraction=prune_fraction)
        
        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        return loss, sample_size, logging_output
    
    def get_lprobs_and_target(self, model, net_output, sample, length_prune=0):

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        
        assert length_prune <= 1, "length_prune must be smaller than 1"

        if length_prune > 0:
            if getattr(lprobs, "batch_first", False):
                seq_len = lprobs.shape[1]
            else:
                seq_len = lprobs.shape[0]

            self.ignore_prefix_size = int(seq_len * length_prune)
      

        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1), lprobs, target

    def compute_loss(self, model, net_output, sample, reduce=True, threshold=None, metric='el2n', length_prune=0, ignore_batch_indices=None, prune_fraction=0):
        
        #computes the loss but ignores the tokens with large el2n scores for a given percentage (hard examples)
        lprobs, target, batched_lprobs, batched_target = self.get_lprobs_and_target(model, net_output, sample, length_prune=length_prune)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            batched_lprobs,
            target,
            batched_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            threshold=threshold,
            threshold_type=metric,
            ignore_batch_indices=ignore_batch_indices,
            prune_fraction=prune_fraction
        )
        
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
