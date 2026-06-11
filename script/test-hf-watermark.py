import builtins
import sys
import types

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from transformers import WatermarkingConfig, WatermarkLogitsProcessor, WatermarkDetector
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


class MyWLP(WatermarkLogitsProcessor):
    def __init__(
            self, vocab_size,
            device,
            greenlist_ratio: float = 0.25,
            bias: float = 2.0,
            hashing_key: int = 15485863,
            seeding_scheme: str = "lefthash",
            context_width: int = 1,
    ):
        super().__init__(vocab_size, device, greenlist_ratio, bias, hashing_key, seeding_scheme, context_width)
        self.ciwater = 'alfred_watson'

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.context_width:
            logger.warning(
                f"`input_ids` should have at least `{self.context_width}` tokens but has {input_ids.shape[-1]}. "
                "The seeding will be skipped for this generation step!"
            )
            return scores
        print("=" * 30)
        scores_processed = scores.clone()
        for b_idx, input_seq in enumerate(input_ids):
            if self.seeding_scheme == "selfhash":
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            scores_processed[b_idx, greenlist_ids] = scores_processed[b_idx, greenlist_ids] + self.bias
        print("=" * 30)
        return scores_processed


class MyWLP2(LogitsProcessor):
    def __init__(
            self,
            greenlist_ratio: float = 0.25,
            bias: float = 2.0,
            hashing_key: int = 15485863,
            seeding_scheme: str = "lefthash",
            context_width: int = 1,
    ):
        self.greenlist_ratio = greenlist_ratio
        self.bias = bias
        self.hashing_key = hashing_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width
        self.ciwater = 'alfred_watson'

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.context_width:
            logger.warning(
                f"`input_ids` should have at least `{self.context_width}` tokens but has {input_ids.shape[-1]}. "
                "The seeding will be skipped for this generation step!"
            )
            return scores
        print("=" * 30)
        scores_processed = scores.clone()
        for b_idx, input_seq in enumerate(input_ids):
            if self.seeding_scheme == "selfhash":
                print("gooood")
            else:
                print("niiice")
        print("=" * 30)
        return scores_processed


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "/home/haojifei/dev_resource/huggingface/models/facebook/opt-1.3b",
        device_map='auto'
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("/home/haojifei/dev_resource/huggingface/models/facebook/opt-1.3b")
    inputs = tokenizer(["Alice and Bob are"], return_tensors="pt").to(model.device)

    # normal generation
    # out = model.generate(inputs["input_ids"], max_length=20, do_sample=False)
    # text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    # >> 'Alice and Bob are both in the same room.\n\n"I\'m not sure if you\'re'

    # watermarked generation
    # watermarking_config = WatermarkingConfig(bias=2.0, context_width=1, seeding_scheme="lefthash")
    out = model.generate(
        inputs["input_ids"],
        logits_processor=[MyWLP2()],
        # watermarking_config=watermarking_config,
        max_length=20, do_sample=False
    )
    text2 = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    # >> 'Alice and Bob are both still alive and well and the story is pretty much a one-hour adventure'

    # to detect watermarked text use the WatermarkDetector class

    detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
    detection_preds = detector(out)
    print(detection_preds)


if __name__ == '__main__':
    # module_name = WatermarkLogitsProcessor.__module__
    # setattr(sys.modules[module_name], 'WatermarkLogitsProcessor', MyWLP)
    main()
