# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from collections import defaultdict

@dataclass
class LabelSmoothedCrossEntropyIwsltCriterionConfig(FairseqDataclass):
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


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
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
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_iwslt", dataclass=LabelSmoothedCrossEntropyIwsltCriterionConfig
)
class LabelSmoothedCrossEntropyIwsltCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        
        # self.idx2vocab = defaultdict(str)
        # self.vocab2idx = defaultdict(int)
        
        # Added Jun 5 - Consturcting a dictionary
        # with open('/scratch4/danielk/tli104/shared_dict.txt', 'r') as f:
        #    for line in f:
        #        line = line.strip().split()
        #        self.idx2vocab[int(line[0])] = line[1]
        #        self.vocab2idx[line[1]] = int(line[0])
        

    def forward(self, model, sample, reduce=True, print_scores=True):
        
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce, print_scores=print_scores)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        return loss, sample_size, logging_output
    
    def compute_el2n(self, probs, target, log_probs=False):
        
        # Parameters:
        # Probs: torch.Tensor, shape of (batch size, length, vocab size)
        # Target: torch.Tensor, shape of (batch size, length)
        # log_probs: boolean, indicating whether the input probs tensor is log probs or raw probs 

        # Returns:
        # The 2 norm of the error vector el2n: 
        # torch.Tensor, shape of (batch size, length)
        
        if log_probs:
            probs = torch.exp(probs)
        
        vocab_size = probs.size(-1)
        batch_size = probs.size(0)
        target_one_hot = F.one_hot(target, vocab_size)
        # masking out the <pad> tokens
        mask = target == 1
        target_one_hot[:, :, 1] = 0 
        probs[mask, :] = torch.zeros(vocab_size).to(device='cuda:0')
        
        return torch.linalg.norm(probs-target_one_hot, dim=-1)

    def get_lprobs_and_target(self, model, net_output, sample, print_scores=None):
        
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)

        # shape of lprobs = batch size, sentence length, vocabulary
        # shape of target = batch size, sentence length
        # print sentence and its log prob to a file 
        # print(sample["net_input"]["src_tokens"])
        
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        
        if print_scores == None:
            return lprobs.view(-1, lprobs.size(-1)), target.view(-1)
        
        probs = model.get_normalized_probs(net_output, log_probs=False)
        prob_of_correct = torch.gather(probs, 2, target.unsqueeze(2)).squeeze(2)
        
        list_of_sources = []
        list_of_targets = []

        for sentence in sample["net_input"]["src_tokens"]:
            sentence = sentence.tolist()
            src_sent = ""
            for index in sentence:
                src_sent += str(index) + ' ' 
            list_of_sources.append(src_sent)
        
        for sentence in target:
            sentence = sentence.tolist()
            tgt_sent = ""
            for index in sentence:
                tgt_sent += str(index) + ' '
            list_of_targets.append(tgt_sent)
        
        el2n = self.compute_el2n(probs, target)
        avg_el2n = torch.mean(el2n, dim=-1)

        with open(f"{print_scores}", 'a+') as f:
            for index, pair in enumerate(zip(list_of_sources, list_of_targets)):
                print(pair[0], file=f)
                print(pair[1], file=f)
                print(' '.join(str(x) for x in el2n[index, :].tolist()), file=f)
                print(' '.join(str(x) for x in prob_of_correct[index, :].tolist()), file=f)

        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, print_scores=None):
        
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, print_scores=print_scores)
    
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
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
