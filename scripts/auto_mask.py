from PIL import Image
from rembg import remove, new_session
import numpy as np

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed
from modules.shared import opts, cmd_opts, state
import modules.scripts as scripts
import gradio as gr


def import_or_install(package,pip_name=None):
    import importlib
    import subprocess
    if pip_name is None:
        pip_name=package
    try:
        importlib.import_module(package)
        print(f"{package} is already installed")
    except ImportError:
        print(f"{package} is not installed, installing now...")
        subprocess.call(['pip', 'install', package])
        print(f"{package} has been installed")


def remove_background(image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,\
                      session_name,only_mask,post_process_mask):
    import_or_install("rembg","rembg[gpu]")
    session=new_session(session_name)
    return remove(image,
                alpha_matting, 
                alpha_matting_foreground_threshold, 
                alpha_matting_background_threshold, 
                alpha_matting_erode_size, 
                session, 
                only_mask,
                post_process_mask)
class Script(scripts.Script):
    def title(self):
        return "Auto Mask"
    def show(self, is_img2img):
        return is_img2img
    def ui(self, is_img2img):
        if not is_img2img: return
        alpha_matting=gr.inputs.Checkbox(label="Alpha Matting")
        alpha_matting_foreground_threshold=gr.inputs.Slider(minimum=0, maximum=255,step=1, default=240, label="Alpha Matting Foreground Threshold")
        alpha_matting_background_threshold=gr.inputs.Slider(minimum=0, maximum=255,step=1, default=10, label="Alpha Matting Background Threshold")
        alpha_matting_erode_size=gr.inputs.Slider(minimum=0, maximum=255,step=1, default=10, label="Alpha Matting Erode Size")
        session_name=gr.inputs.Dropdown(["u2net", "u2netp","u2net_human_seg","u2net_cloth_seg","silueta"], label="Session")
        only_mask=gr.inputs.Checkbox(label="Only Mask")
        post_process_mask=gr.inputs.Checkbox(label="Post Process Mask")
        with gr.Blocks() as demo:
            with gr.Row().style(equal_height=True):
                image=gr.Image(type="pil")
                mask=gr.Image(type="pil")
        btn = gr.Button(value="Preview Remove Background")
        if image is not None:
            btn.click(remove_background, inputs=[image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,\
                                                 alpha_matting_erode_size,session_name,only_mask,post_process_mask], outputs=[mask])
        return [image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,session_name,\
                only_mask,post_process_mask]
    def run(self,p,image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,session_name,\
                only_mask,post_process_mask):
        if image is None:
            image=p.init_images[0]
        only_mask=True
        mask=remove_background(image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,\
                                                 alpha_matting_erode_size,session_name,only_mask,post_process_mask)
        p.image_mask=mask
        proc = process_images(p)
        proc.images.append(mask)
        return proc
