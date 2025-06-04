# ==================== SETUP ====================
!git clone
%cd shap-e
!pip install -e .
!pip install -q gradio trimesh imageio pillow

# ==================== IMPORTS ====================
import os
import torch
import gradio as gr
import imageio
import trimesh
import numpy as np
from PIL import Image, ImageDraw
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

# ==================== DEVICE & MODEL LOADING ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load models
try:
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    xm = load_model('transmitter', device=device)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = diffusion = xm = None

# ==================== IMPROVED GIF CREATION ====================
def create_rotating_gif(obj_path, gif_path, frames=20, size=400):
    """Create a rotating GIF with proper colors and materials"""
    try:
        print("📷 Creating rotating GIF...")

        # Load mesh
        mesh = trimesh.load(obj_path)
        if mesh.is_empty:
            print("❌ Empty mesh loaded")
            return False

        # Center and scale mesh
        mesh.apply_translation(-mesh.centroid)
        scale = 2.0 / mesh.scale  # Make it bigger
        mesh.apply_scale(scale)

        # Add some color if mesh doesn't have vertex colors
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            # Create gradient colors based on vertex height
            vertices = mesh.vertices
            min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
            if max_y > min_y:
                # Create height-based color gradient
                normalized_height = (vertices[:, 1] - min_y) / (max_y - min_y)
                colors = np.zeros((len(vertices), 4))
                colors[:, 0] = 0.2 + 0.6 * normalized_height  # Red component
                colors[:, 1] = 0.3 + 0.4 * (1 - normalized_height)  # Green component
                colors[:, 2] = 0.6 + 0.4 * normalized_height  # Blue component
                colors[:, 3] = 1.0  # Alpha
                colors = (colors * 255).astype(np.uint8)
                mesh.visual.vertex_colors = colors
            else:
                # Default color if mesh is flat
                mesh.visual.face_colors = [100, 150, 200, 255]

        frames_list = []

        for i in range(frames):
            print(f"🎬 Rendering frame {i+1}/{frames}")

            # Create rotation angle
            angle = (i / frames) * 2 * np.pi

            # Create rotation matrix around Y axis
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])

            # Apply rotation
            rotated_mesh = mesh.copy()
            rotated_mesh.apply_transform(rotation_matrix)

            # Create scene with better lighting
            scene = trimesh.Scene(rotated_mesh)

            # Add multiple lights for better visualization
            camera_transform = np.eye(4)
            camera_transform[2, 3] = 3  # Move camera back

            try:
                # Try to render with pyrender if available
                png = scene.save_image(
                    resolution=[size, size],
                    visible=True,
                    camera_transform=camera_transform
                )

                if png is not None:
                    image = Image.open(trimesh.util.wrap_as_stream(png))
                    # Convert RGBA to RGB if needed
                    if image.mode == 'RGBA':
                        # Create white background
                        white_bg = Image.new('RGB', image.size, (255, 255, 255))
                        white_bg.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                        image = white_bg
                    frames_list.append(np.array(image))
                else:
                    raise Exception("Rendering returned None")

            except Exception as render_error:
                print(f"Render error: {render_error}, using fallback")
                # Enhanced wireframe fallback with colors
                image = create_enhanced_wireframe(rotated_mesh, size)
                frames_list.append(np.array(image))

        if frames_list:
            # Save GIF with better settings
            imageio.mimsave(
                gif_path,
                frames_list,
                fps=10,  # Smoother animation
                loop=0,
                duration=0.10
            )
            print(f"✅ GIF saved: {gif_path}")
            return True
        else:
            print("❌ No frames created")
            return False

    except Exception as e:
        print(f"❌ GIF creation error: {e}")
        return False

def create_enhanced_wireframe(mesh, size=400):
    """Create an enhanced wireframe with better colors and shading"""
    try:
        # Create image with gradient background
        img = Image.new('RGB', (size, size), color=(240, 240, 250))
        draw = ImageDraw.Draw(img)

        # Add subtle gradient background
        for y in range(size):
            color_val = int(240 + (y / size) * 15)
            draw.line([(0, y), (size, y)], fill=(color_val, color_val, color_val + 10))

        vertices = mesh.vertices

        if len(vertices) > 0:
            # Project vertices to 2D
            min_vals = vertices.min(axis=0)
            max_vals = vertices.max(axis=0)
            range_vals = max_vals - min_vals

            if range_vals.max() > 0:
                # Project with better centering
                padding = 0.15
                scale_factor = (1 - 2 * padding)

                x_coords = ((vertices[:, 0] - min_vals[0]) / range_vals.max() * size * scale_factor + size * padding).astype(int)
                y_coords = ((vertices[:, 1] - min_vals[1]) / range_vals.max() * size * scale_factor + size * padding).astype(int)

                # Use Z coordinate for depth-based coloring
                z_coords = vertices[:, 2]
                z_min, z_max = z_coords.min(), z_coords.max()

                # Draw faces with depth-based colors
                if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                    faces = mesh.faces[:min(200, len(mesh.faces))]

                    for face in faces:
                        face_vertices = []
                        face_z = []

                        for vertex_idx in face:
                            if vertex_idx < len(x_coords):
                                face_vertices.append((x_coords[vertex_idx], y_coords[vertex_idx]))
                                face_z.append(z_coords[vertex_idx])

                        if len(face_vertices) == 3:
                            # Calculate face depth
                            avg_z = np.mean(face_z)
                            if z_max > z_min:
                                depth_ratio = (avg_z - z_min) / (z_max - z_min)
                            else:
                                depth_ratio = 0.5

                            # Color based on depth (closer = brighter)
                            base_color = int(80 + depth_ratio * 100)
                            face_color = (base_color, base_color + 20, base_color + 40)
                            edge_color = (max(0, base_color - 30), max(0, base_color - 10), base_color + 60)

                            # Draw filled triangle
                            try:
                                draw.polygon(face_vertices, fill=face_color, outline=edge_color, width=1)
                            except:
                                # Fallback to lines if polygon fails
                                for i in range(3):
                                    draw.line([face_vertices[i], face_vertices[(i+1)%3]], fill=edge_color, width=2)

                # Draw vertices as points
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    if 0 <= x < size and 0 <= y < size:
                        z = z_coords[i]
                        if z_max > z_min:
                            depth_ratio = (z - z_min) / (z_max - z_min)
                        else:
                            depth_ratio = 0.5

                        point_color = (int(100 + depth_ratio * 155), int(50 + depth_ratio * 100), int(200 + depth_ratio * 55))
                        draw.ellipse([x-2, y-2, x+2, y+2], fill=point_color)

        # Add title
        draw.text((10, 10), "3D Model Preview", fill=(60, 60, 60))

        return img

    except Exception as e:
        print(f"Enhanced wireframe error: {e}")
        # Simple fallback
        img = Image.new('RGB', (size, size), color=(220, 220, 230))
        draw = ImageDraw.Draw(img)
        draw.text((size//2-80, size//2), "3D Model Generated", fill=(100, 100, 100))
        return img

# ==================== GENERATE 3D MODEL FUNCTION ====================
def generate_3d_model(prompt, guidance_scale=15.0, steps=64):
    """Generate 3D model from text prompt"""
    if not prompt.strip():
        return None, None, "⚠️ Please enter a prompt"

    if model is None or diffusion is None or xm is None:
        return None, None, "❌ Models not loaded"

    try:
        print(f"🚀 Generating: '{prompt}'")

        # Generate latents
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            device=device,
            use_karras=True,
            karras_steps=steps,
            sigma_min=0.002,
            sigma_max=160,
            s_churn=0
        )

        # Create output directory
        os.makedirs("outputs", exist_ok=True)

        # Decode mesh
        print("🔧 Decoding mesh...")
        mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()

        # Save OBJ
        obj_path = "outputs/model.obj"
        with open(obj_path, "w") as f:
            mesh.write_obj(f)
        print(f"💾 Saved: {obj_path}")

        # Create GIF
        gif_path = "outputs/preview.gif"
        gif_success = create_rotating_gif(obj_path, gif_path)

        if gif_success:
            success_msg = f"✅ <strong>Success!</strong> Generated 3D model for: '<em>{prompt}</em>'"
            return obj_path, gif_path, success_msg
        else:
            success_msg = f"✅ <strong>Generated!</strong> 3D model for: '<em>{prompt}</em>' <br>⚠️ <small>GIF preview failed - check OBJ file</small>"
            return obj_path, None, success_msg

    except Exception as e:
        error_msg = f"❌ <strong>Generation Failed:</strong> {str(e)}"
        print(error_msg)
        return None, None, error_msg

# ==================== GRADIO INTERFACE ====================
examples = [
    "A birthday cupcake",
    "A penguin",
    "A green boot",
    "A chair that looks like an avocado",
    "An airplane that looks like a banana",
    "A spaceship"
]

# Custom CSS for better styling
css = """
.status-box {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.preview-container {
    border: 1px solid #d0d0d0;
    border-radius: 10px;
    padding: 10px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
/* Remove border from image preview */
.gr-image {
    border: none !important;
    box-shadow: none !important;
}
"""


with gr.Blocks(title="SHAP-E 3D Generator", css=css) as demo:

    gr.Markdown("# SHAP-E Transforming Descriptive Text into  3D Models")
    gr.Markdown("**Create detailed 3D models just by typing what you want. No special skills needed SHAP-E turns your words into digital 3D objects quickly and easily.**")

    with gr.Row():
        # Left Column - Input Controls
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Model Description")
            prompt = gr.Textbox(
                label="Describe your 3D model",
                lines=3,
                max_lines=5
            )

            # Quick Examples
            gr.Markdown("**💡 Quick Examples:**")
            example_buttons = []
            for example in examples:
                btn = gr.Button(example, size="sm")
                btn.click(lambda x=example: x, outputs=prompt)

            # Advanced Settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                guidance_scale = gr.Slider(
                    minimum=5,
                    maximum=25,
                    value=15,
                    step=1,
                    label="Guidance Scale (higher = more prompt adherence)"
                )
                steps = gr.Slider(
                    minimum=32,
                    maximum=128,
                    value=64,
                    step=8,
                    label="Generation Steps (higher = better quality)"
                )

            # Generate Button
            generate_btn = gr.Button(
                "🚀 Generate 3D Model",
                variant="primary",
                size="lg"
            )

            # Status Display (moved here under the button)
            gr.Markdown("### 📊 Generation Status")
            status = gr.HTML(
                value="<div style='padding: 10px; text-align: center; color: #666;'>Ready to generate! Enter a description above.</div>",
                elem_classes=["status-box"]
            )

        # Right Column - Preview and Download
        with gr.Column(scale=1.5):
            # 3D Preview Section
            gr.Markdown("### 🎬 3D Model Preview")
            with gr.Group(elem_classes=["preview-container"]):
                preview_gif = gr.Image(
                    label="360° Rotating View",
                    type="filepath",
                    height=350,
                    show_label=True,
                    container=True
                )
                gr.Markdown("<small>*The model will rotate automatically once generated*</small>")

            # Download Section
            gr.Markdown("### 💾 Download Files")
            obj_file = gr.File(
                label="📄 3D Model File (.obj)",
                file_count="single"
            )
            gr.Markdown("<small>*Download the .obj file to use in 3D software like Blender*</small>")

    # Event Handlers
    def generate_and_update(prompt_text, guidance_scale, steps):
        if not prompt_text.strip():
            return (
                "<div style='color: orange; padding: 10px;'>⚠️ <strong>Please enter a description</strong> for your 3D model</div>",
                None,
                None
            )

        # Show generating status
        generating_status = f"<div style='color: blue; padding: 10px;'>🔄 <strong>Generating...</strong> Creating 3D model for: '<em>{prompt_text}</em>'<br><small>This may take 1-2 minutes...</small></div>"

        obj_path, gif_path, message = generate_3d_model(prompt_text, guidance_scale, steps)

        # Format the final status message
        if "Success" in message:
            final_status = f"<div style='color: green; padding: 10px;'>{message}</div>"
        elif "Generated" in message:
            final_status = f"<div style='color: green; padding: 10px;'>{message}</div>"
        else:
            final_status = f"<div style='color: red; padding: 10px;'>{message}</div>"

        return final_status, gif_path, obj_path

    # Connect the generate button
    generate_btn.click(
        fn=generate_and_update,
        inputs=[prompt, guidance_scale, steps],
        outputs=[status, preview_gif, obj_file]
    )


# ==================== LAUNCH ====================
print("🚀 Launching Improved SHAP-E Generator...")
demo.launch(share=True, show_error=True)