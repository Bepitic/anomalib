"""HuggingFace Wrapper for VLM  Zero-/Few-Shot Anomaly Classification.

Paper No paper
"""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from io import BytesIO

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize, Transform
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

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
        model_path: str = "llava-hf/llava-v1.6-mistral-7b-hf",
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

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", load_in_4bit=self.load4bits
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _setup(self) -> None:
        dataloader = self.trainer.datamodule.train_dataloader()
        pre_images = self.collect_reference_images(dataloader)
        self.pre_images = pre_images

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
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
                o = str(self.api_call_fewShot(self.pre_images, "", batch["image_path"][x])).strip()
            else:
                o = str(self.api_call("", batch["image_path"][x])).strip()
            p = 0.0 if o.startswith("N") else 1.0
            out_list.append(o)
            pred_list.append(p)

        batch["str_output"] = out_list
        batch["pred_scores"] = torch.tensor(pred_list).to(self.model.device)
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

    def load_image(self, image_file: str) -> Image:
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        """Configure the default transforms used by the model."""
        if image_size is not None:
            logger.warning("Image size is not used in WinCLIP. The input image size is determined by the model.")
        return Compose(
            [
                Resize((520, 520), antialias=True, interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )

    def api_call_fewShot(self, pre_images: str, prompt: str, image_path: str) -> str:
        images = []
        images_size = []

        for img_path in pre_images:
            i = self.load_image(img_path)
            images_size.append(i.size)
            images.append(i)

        img = self.load_image(image_path)
        prompt = ""
        preprompt = ""

        promptend = "From this 2 images, the first one being a normal image, and the second one a possibly abnormal one Check if the second one diverges in an obvious abnormal form from the first one and report if there is an abnormality,  If the Object contains any defects, irregularities, or anomalies, respond with 'YES:description' where 'description' explains the specific defect(s) found, if there is not a defect then say NO, and stop."

        # Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
        conversation_1 = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": promptend},
                    {"type": "image"},
                    {"type": "image"},
                ],
            },
        ]

        prompt_1 = self.processor.apply_chat_template(conversation_1, add_generation_prompt=True)
        prompts = [prompt_1]

        # We can simply feed images in the order they have to be used in the text prompt
        # Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
        inputs = self.processor(text=prompts, images=[images[0], img], padding=False, return_tensors="pt").to(
            self.model.device
        )

        # Generate
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=300, pad_token_id=self.processor.tokenizer.pad_token_id
        )
        text_outputs = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        parts = text_outputs[0].split("[/INST]")

        return parts[1].strip()
