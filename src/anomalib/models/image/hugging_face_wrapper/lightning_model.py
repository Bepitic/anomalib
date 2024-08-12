"""HuggingFace Wrapper for VLM  Zero-/Few-Shot Anomaly Classification.

Paper No paper
"""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)

__all__ = ["HuggingFaceWrapper"]


class HuggingFaceWrapper(AnomalyModule):
    """HuggingFace Wrapper Lightning model.

    This class serves as a wrapper for integrating HuggingFace models within
    a PyTorch Lightning framework, specifically tailored for anomaly detection
    tasks.

    Args:
        k_shot (int, optional): Number of shots (examples) to use for few-shot learning.
            Defaults to 0.
        temperature (float, optional): Temperature value to control the randomness of predictions.
            A higher value makes the model's output more random, while a lower value makes it more deterministic.
            Defaults to 0.0.
        model_path (str, optional): Path or identifier of the model to load from HuggingFace.
            Defaults to "llava-hf/llava-v1.6-mistral-7b-hf".
        load_8bits (bool, optional): Flag to indicate whether to load the model in 8-bit precision.
            Useful for reducing memory usage. Defaults to False.
        load_4bits (bool, optional): Flag to indicate whether to load the model in 4-bit precision.
            Further reduces memory usage compared to 8-bit precision. Defaults to False.
        max_new_tokens (int, optional): Maximum number of new tokens to generate during inference.
            This limits the length of the generated sequence. Defaults to 100, with a possible maximum of 1024.

    """

    def __init__(
        self,
        k_shot: int = 0,
        temperature: float = 0.0,
        model_path: str = "llava-hf/llava-interleave-qwen-7b-hf",
        load_8bits: bool = False,
        load_4bits: bool = False,
        max_new_tokens: int = 100,  # max 1024
    ) -> None:
        super().__init__()
        self.k_shot = k_shot
        self.temperature = temperature
        self.load8bits = load_8bits
        self.load4bits = load_4bits
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.pre_images: list[str] = []

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=self.load4bits,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _setup(self) -> None:
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict[str, str | torch.Tensor]:
        """Train Step of LLM."""
        del args, kwargs  # These variables are not used.
        # no train on llm
        return batch

    @staticmethod
    def configure_optimizers() -> None:
        """WinCLIP doesn't require optimization, therefore returns no optimizers."""
        return

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict:
        """Validation Step of WinCLIP."""
        self._setup()
        del args, kwargs  # These variables are not used.
        bsize = len(batch["image_path"])
        out_list: list[str] = []
        pred_list = []
        for x in range(bsize):
            o = "NO - default"
            if self.k_shot > 0:
                o = self._api_call_few_shot(batch["image_path"][x])
            else:
                o = self._api_call_zero_shot(batch["image_path"][x])
            p = 0.0 if o.startswith("N") else 1.0
            out_list.append(o)
            pred_list.append(p)

        batch["str_output"] = out_list
        batch["pred_scores"] = torch.tensor(pred_list).to(self.device)
        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Set model-specific trainer arguments."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """The learning type of the model.

        Llm is a zero-/few-shot model, depending on the user configuration. Therefore, the learning type is
        set to ``LearningType.FEW_SHOT`` when ``k_shot`` is greater than zero and ``LearningType.ZERO_SHOT`` otherwise.
        """
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

    def collect_reference_images(self, dataloader: DataLoader) -> list[str]:
        """Collect reference images for few-shot inference.

        The reference images are collected by iterating the training dataset until the required number of images are
        collected.

        Returns:
            ref_images (Tensor): A tensor containing the reference images.
        """
        ref_images: list[str] = []
        for batch in dataloader:
            images = batch["image_path"][: self.k_shot - len(ref_images)]
            ref_images.extend(images)
            if self.k_shot == len(ref_images):
                break
        return ref_images

    def _load_image(self, image_file: str) -> Image.Image:
        return Image.open(image_file).convert("RGB")

    def _api_call_zero_shot(self, image_path: str) -> str:
        img = self._load_image(image_path)

        prompt = """
        Examine the provided image carefully to determine if there is an obvious anomaly present.
        Anomalies may include mechanical malfunctions, unexpected objects, safety hazards, structural damages,
        or unusual patterns or defects in the objects.

        Instructions:

        1. Thoroughly inspect the image for any irregularities or deviations from normal operating conditions.

        2. Clearly state if an obvious anomaly is detected.
        - If an anomaly is detected, begin with 'YES,' followed by a detailed description of the anomaly.
        - If no anomaly is detected, simply state 'NO' and end the analysis.

        Example Output Structure:

        'YES:
        - Description: Conveyor belt misalignment causing potential blockages.
        This may result in production delays and equipment damage.
        Immediate realignment and inspection are recommended.'

        'NO'

        Considerations:

        - Ensure accuracy in identifying anomalies to prevent overlooking critical issues.
        - Provide clear and concise descriptions for any detected anomalies.
        - Focus on obvious anomalies that could impact final use of the object operation or safety.
        """

        # Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
        conversation_1 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        prompt_1 = self.processor.apply_chat_template(conversation_1, add_generation_prompt=True)
        inputs = self.processor(prompt_1, img, return_tensors="pt").to(self.device, torch.float16)
        # Obtaining the len of the prompt in tokens.
        token_len = inputs.input_ids.shape[1]
        print("inputs")
        print(type(inputs))
        for key, value in inputs.items():
            print(f"{key}: {value.shape}")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        generated_outputs = outputs[:, token_len:]
        text_outputs = self.processor.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return text_outputs[0]

    def _api_call_few_shot(self, image_path: str) -> str:
        images = []
        i = self._load_image(image_path)
        images.append(i)
        for img_path in self.pre_images:
            i = self._load_image(img_path)
            images.append(i)

        prompt = """
You will receive a group of images that is going to be an example of the typical image without any anomaly,
and the last image that you need to decide if it has an anomaly or not.
Answer with a 'NO' if it does not have any anomalies and 'YES: description'
where description is a description of the anomaly provided, position.
"""

        # Start with the text prompt
        content = [{"type": "text", "text": prompt}]
        content.extend([{"type": "image"} for _ in range(len(images))])

        # Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
        conversation_1 = [
            {
                "role": "user",
                "content": content,
            },
        ]

        prompt_1 = self.processor.apply_chat_template(conversation_1, add_generation_prompt=True)
        inputs = self.processor(prompt_1, images, return_tensors="pt").to(self.device, torch.float16)
        # Obtaining the len of the prompt in tokens.
        token_len = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        generated_outputs = outputs[:, token_len:]
        text_outputs = self.processor.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return text_outputs[0]
