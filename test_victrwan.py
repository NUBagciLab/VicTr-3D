# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, VicTrWanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel
import nibibal as nib

text_encoder = UMT5EncoderModel.from_pretrained("Onkarsus13/VicTr-3D", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Onkarsus13/VicTr-3D", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Onkarsus13/VicTr-3D", subfolder="transformer", torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
apply_group_offloading(text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)
transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)

pipeline = VicTrWanPipeline.from_pretrained(
    "Onkarsus13/VicTr3D",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
mask = nib.load("<Mask.nii.gz>")
 
output = pipeline(
    mask = mask,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)