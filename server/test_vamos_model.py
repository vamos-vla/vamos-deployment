#!/usr/bin/env python3
"""
Simple test script to run an image through the VAMOS model.
This script tests the VLM navigation functionality with a sample image.
"""

import os
import sys
import argparse
import base64
import io
import time
import glob
import re
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftConfig, PeftModel

def create_test_image(width=640, height=480, scenario="indoor"):
    """
    Create a test navigation image.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        scenario: Type of scenario ('indoor', 'outdoor', 'office')
    
    Returns:
        PIL Image object
    """
    # Create a blank image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    if scenario == "indoor":
        # Draw walls and obstacles
        # Left wall
        draw.rectangle([0, 0, 50, height], fill='gray')
        # Right wall
        draw.rectangle([width-50, 0, width, height], fill='gray')
        # Back wall
        draw.rectangle([0, 0, width, 50], fill='gray')
        
        # Add some furniture/obstacles
        # Table in the middle
        draw.rectangle([width//2-60, height//2-40, width//2+60, height//2+40], fill='brown')
        # Chair
        draw.rectangle([width//2+80, height//2-20, width//2+100, height//2+20], fill='brown')
        
        # Add a door
        draw.rectangle([width-100, height//2-30, width-50, height//2+30], fill='darkgreen')
        
    elif scenario == "outdoor":
        # Draw a simple outdoor scene
        # Sky (top half)
        draw.rectangle([0, 0, width, height//2], fill='lightblue')
        # Ground (bottom half)
        draw.rectangle([0, height//2, width, height], fill='lightgreen')
        
        # Add some trees
        for x in [100, 200, 400, 500]:
            # Tree trunk
            draw.rectangle([x-5, height//2, x+5, height-50], fill='brown')
            # Tree canopy
            draw.ellipse([x-20, height//2-20, x+20, height//2+20], fill='darkgreen')
        
        # Add a path
        draw.rectangle([0, height//2+20, width, height//2+40], fill='tan')
        
    elif scenario == "office":
        # Draw an office environment
        # Walls
        draw.rectangle([0, 0, 30, height], fill='lightgray')
        draw.rectangle([width-30, 0, width, height], fill='lightgray')
        draw.rectangle([0, 0, width, 30], fill='lightgray')
        
        # Desks
        draw.rectangle([100, 100, 200, 150], fill='brown')
        draw.rectangle([300, 200, 400, 250], fill='brown')
        
        # Chairs
        draw.rectangle([120, 80, 140, 100], fill='black')
        draw.rectangle([320, 180, 340, 200], fill='black')
        
        # Door
        draw.rectangle([width-80, height//2-40, width-30, height//2+40], fill='darkgreen')
    
    return img

def load_vamos_model(model_name_or_path, use_fp32=False, trust_remote_code=True):
    """
    Load the VAMOS model and processor from Hugging Face.
    
    Args:
        model_name_or_path: Hugging Face model name or local path
        use_fp32: Use FP32 precision instead of bfloat16
        trust_remote_code: Trust remote code for model loading
    
    Returns:
        tuple: (model, processor, device) or (None, None, None) if failed
    """
    print(f"🔄 Loading VAMOS model from: {model_name_or_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    try:
        if "lora" in model_name_or_path.lower():
            print("📦 Loading LoRA model...")
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load processor from base model
            processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Load base model first
            model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32 if use_fp32 else torch.bfloat16,
                trust_remote_code=trust_remote_code
            ).to(device)
            
            # Then load LoRA adapter
            model = PeftModel.from_pretrained(model, model_name_or_path)
            
            print("✅ LoRA model loaded successfully!")
            return model, processor, device
        else:
            print("📦 Loading standard model...")
            # Load regular model
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code
            )
            
            model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32 if use_fp32 else torch.bfloat16,
                trust_remote_code=trust_remote_code
            ).to(device)
            
            print("✅ Model loaded successfully!")
            return model, processor, device
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def decode_trajectory_string(trajectory_string, img_height, img_width, expected_tokens=10):
    """Convert a string of location tokens into a list of coordinates."""
    # Use regex to find all <loc> tokens
    generated_traj = trajectory_string.split("\n")[-1]
    tokens = re.findall(r'<loc\d{4}>', generated_traj)
    
    if len(tokens) < 2:
        return torch.empty((0, 2), dtype=torch.float32)
    
    # Convert tokens to coordinates
    coords = []
    for token in tokens:
        num = int(token[4:8])
        coords.append(num)

    # Padding logic
    if len(coords) < expected_tokens:
        if len(coords) % 2 != 0:
            coords = coords[:-1]
            
        if len(coords) >= 2:
            last_pair = coords[-2:]
            while len(coords) < expected_tokens:
                coords.extend(last_pair)
        elif len(coords) == 0:
             return torch.empty((0, 2), dtype=torch.float32)

    if not coords:
        return torch.empty((0, 2), dtype=torch.float32)

    coords_tensor = torch.tensor(coords).reshape(-1, 2).float()
    
    # Normalize coordinates back to image space
    if img_width > 0:
        coords_tensor[:, 0] = coords_tensor[:, 0] * img_width / 1024
    else:
         coords_tensor[:, 0] = 0
    if img_height > 0:
        coords_tensor[:, 1] = coords_tensor[:, 1] * img_height / 1024
    else:
         coords_tensor[:, 1] = 0
    
    return coords_tensor

def generate_trajectories(model, processor, image, text_prompt, max_tokens=10, temperature=0.1, 
                         top_k=0, num_beams=1, num_samples=3, device="cpu"):
    """
    Generate trajectories using the loaded model.
    
    Args:
        model: Loaded VLM model
        processor: Loaded processor
        image: PIL Image object
        text_prompt: Navigation prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search
        num_samples: Number of samples to generate
        device: Device to run inference on
    
    Returns:
        tuple: (trajectories, generated_texts, error_message)
    """
    try:
        print(f"🧠 Generating trajectories with prompt: '{text_prompt}'")
        
        # Prepare model input
        model_input = processor(
            text=["<image><bos>" + text_prompt],
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding="longest",
        ).to(device)
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": (temperature > 0),
            "num_return_sequences": num_samples
        }
        
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
        if num_beams > 1:
            generation_kwargs["num_beams"] = num_beams
        
        # Generate trajectories
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            outputs = model.generate(**model_input, **generation_kwargs)
        
        # Decode output sequences
        generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
        
        all_pred_coords = []
        valid_generated_texts = []
        
        # Process each generated sequence
        for text in generated_texts:
            try:
                pred_coords = decode_trajectory_string(
                    text, 
                    image.size[1],  # height
                    image.size[0],  # width
                    expected_tokens=max_tokens
                )
                if pred_coords.numel() > 0:
                    all_pred_coords.append(pred_coords.tolist())
                    valid_generated_texts.append(text)
            except Exception as e:
                print(f"⚠️  Error decoding trajectory: {e} for text: {text}")
        
        return all_pred_coords, valid_generated_texts, None
        
    except Exception as e:
        error_message = f"Error during model generation: {str(e)}"
        print(f"❌ {error_message}")
        return [], [], error_message

def find_image_files(directory=".", extensions=None):
    """
    Find image files in a directory.
    
    Args:
        directory: Directory to search in
        extensions: List of file extensions to search for
    
    Returns:
        List of found image file paths
    """
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
        # Also search for uppercase extensions
        pattern_upper = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort

def validate_image_path(image_path):
    """
    Validate and resolve image path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        tuple: (is_valid, resolved_path, error_message)
    """
    if not image_path:
        return False, None, "No image path provided"
    
    # Expand user home directory
    expanded_path = os.path.expanduser(image_path)
    
    # Convert to absolute path
    abs_path = os.path.abspath(expanded_path)
    
    if not os.path.exists(abs_path):
        return False, None, f"File does not exist: {abs_path}"
    
    if not os.path.isfile(abs_path):
        return False, None, f"Path is not a file: {abs_path}"
    
    # Check if it's a valid image file
    try:
        with Image.open(abs_path) as img:
            img.verify()
        return True, abs_path, None
    except Exception as e:
        return False, None, f"Invalid image file: {e}"

def test_vamos_model(model_name_or_path, image_path=None, text_prompt="Navigate to the door", 
                    scenario="indoor", max_tokens=10, temperature=0.1, top_k=0, 
                    num_beams=1, num_samples=3, use_fp32=False):
    """
    Test the VAMOS model with an image and text prompt.
    
    Args:
        model_name_or_path: Hugging Face model name or local path
        image_path: Path to custom image (optional)
        text_prompt: Navigation prompt
        scenario: Test scenario for generated image (if no custom image provided)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search
        num_samples: Number of samples to generate
        use_fp32: Use FP32 precision instead of bfloat16
    """
    print("🧭 VAMOS Model Test")
    print("=" * 50)
    
    # Load the model
    model, processor, device = load_vamos_model(model_name_or_path, use_fp32=use_fp32)
    if model is None or processor is None:
        print("❌ Failed to load model. Please check the model path and try again.")
        return False
    
    # Load or create test image
    if image_path:
        print(f"📷 Attempting to load custom image: {image_path}")
        
        # Validate the image path
        is_valid, resolved_path, error_msg = validate_image_path(image_path)
        
        if is_valid:
            try:
                test_image = Image.open(resolved_path).convert('RGB')
                print(f"✅ Successfully loaded image from: {resolved_path}")
            except Exception as e:
                print(f"❌ Error loading image from {resolved_path}: {e}")
                print("📷 Falling back to generated test image...")
                test_image = create_test_image(scenario=scenario)
        else:
            print(f"❌ {error_msg}")
            print("📷 Creating test image instead...")
            test_image = create_test_image(scenario=scenario)
    else:
        print("📷 Creating test image...")
        test_image = create_test_image(scenario=scenario)
    
    print(f"📐 Image size: {test_image.size}")
    
    # Display the test image
    plt.figure(figsize=(10, 6))
    plt.imshow(test_image)
    plt.title('Test Image for VAMOS Navigation')
    plt.axis('off')
    plt.show()
    
    print(f"📝 Testing with prompt: '{text_prompt}'")
    print("🔄 Running inference...")
    
    # Generate trajectories using the loaded model
    start_time = time.time()
    trajectories, generated_texts, error = generate_trajectories(
        model, processor, test_image, text_prompt, 
        max_tokens=max_tokens, temperature=temperature, 
        top_k=top_k, num_beams=num_beams, num_samples=num_samples, device=device
    )
    processing_time = time.time() - start_time
    
    if error:
        print(f"❌ Prediction failed: {error}")
        return False
    
    if trajectories and len(trajectories) > 0:
        print(f"✅ Prediction successful!")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"🎯 Generated {len(trajectories)} trajectories")
        
        # Print generated texts
        for i, text in enumerate(generated_texts):
            print(f"📄 Generated text {i+1}: {text}")
        
        # Visualize results
        visualize_results(test_image, trajectories, text_prompt)
        
        return True
    else:
        print("❌ No valid trajectories generated")
        return False

def visualize_results(image, trajectories, text_prompt):
    """
    Visualize the navigation results.
    
    Args:
        image: PIL Image object
        trajectories: List of trajectory coordinates
        text_prompt: Original text prompt
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Image with trajectories
    ax2.imshow(image)
    
    if trajectories and len(trajectories) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) > 0:
                traj_array = np.array(trajectory)
                if traj_array.shape[1] >= 2:
                    x_coords = traj_array[:, 0]
                    y_coords = traj_array[:, 1]
                    
                    # Plot trajectory
                    ax2.plot(x_coords, y_coords, 'o-', color=colors[i], 
                           linewidth=2, markersize=4, alpha=0.8, 
                           label=f'Trajectory {i+1}')
                    
                    # Mark start and end points
                    ax2.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start' if i == 0 else "")
                    ax2.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End' if i == 0 else "")
    
    ax2.set_title(f'Navigation Results\nPrompt: "{text_prompt}"')
    ax2.axis('off')
    
    if trajectories and len(trajectories) > 0:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print trajectory details
    print("\n📍 Trajectory Details:")
    for i, trajectory in enumerate(trajectories):
        if len(trajectory) > 0:
            traj_array = np.array(trajectory)
            if traj_array.shape[1] >= 2:
                # Calculate trajectory length
                distances = np.sqrt(np.diff(traj_array[:, 0])**2 + np.diff(traj_array[:, 1])**2)
                total_length = np.sum(distances)
                
                print(f"  Trajectory {i+1}: {len(trajectory)} waypoints, length: {total_length:.1f} pixels")
                print(f"    Start: ({traj_array[0, 0]:.1f}, {traj_array[0, 1]:.1f})")
                print(f"    End: ({traj_array[-1, 0]:.1f}, {traj_array[-1, 1]:.1f})")

def main():
    parser = argparse.ArgumentParser(
        description="Test VAMOS model with an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with Hugging Face model
  python test_vamos_model.py --model "mateoguaman/paligemma2-3b-pt-224-sft-lora-magicsoup"

  # Test with local model
  python test_vamos_model.py --model /path/to/local/model

  # Test with custom image
  python test_vamos_model.py --model "model_name" --image /path/to/your/image.jpg

  # Test with custom image and prompt
  python test_vamos_model.py --model "model_name" --image ~/Pictures/room.jpg --prompt "Navigate to the window"

  # Test outdoor scenario
  python test_vamos_model.py --model "model_name" --scenario outdoor --prompt "Follow the path"

  # List available images in current directory
  python test_vamos_model.py --list-images
        """
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Hugging Face model name or local path to model")
    parser.add_argument("--image", type=str, default=None, 
                       help="Path to custom image (supports ~ for home directory)")
    parser.add_argument("--prompt", type=str, default="Navigate to the door", 
                       help="Navigation prompt (default: 'Navigate to the door')")
    parser.add_argument("--scenario", type=str, choices=["indoor", "outdoor", "office"], 
                       default="indoor", help="Test scenario for generated image")
    parser.add_argument("--max-tokens", type=int, default=10,
                       help="Maximum tokens to generate (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature (default: 0.1)")
    parser.add_argument("--top-k", type=int, default=0,
                       help="Top-k sampling parameter (default: 0)")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Number of beams for beam search (default: 1)")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Number of samples to generate (default: 3)")
    parser.add_argument("--use-fp32", action="store_true",
                       help="Use FP32 precision instead of bfloat16")
    parser.add_argument("--list-images", action="store_true",
                       help="List available image files in current directory and exit")
    
    args = parser.parse_args()
    
    # Handle --list-images option
    if args.list_images:
        print("📷 Available image files in current directory:")
        print("=" * 50)
        
        image_files = find_image_files()
        if image_files:
            for i, img_path in enumerate(image_files, 1):
                try:
                    with Image.open(img_path) as img:
                        size = img.size
                        print(f"{i:2d}. {img_path} ({size[0]}x{size[1]})")
                except Exception as e:
                    print(f"{i:2d}. {img_path} (error: {e})")
        else:
            print("No image files found in current directory.")
            print("Supported formats: JPG, JPEG, PNG, BMP, TIFF, GIF")
        
        print("\nUsage examples:")
        print(f"  python {sys.argv[0]} --image image1.jpg")
        print(f"  python {sys.argv[0]} --image ~/Pictures/room.png --prompt 'Navigate to the window'")
        return
    
    print("🚀 Starting VAMOS Model Test")
    print(f"🤖 Model: {args.model}")
    print(f"📝 Prompt: '{args.prompt}'")
    print(f"⚙️  Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, num_samples={args.num_samples}")
    
    if args.image:
        print(f"📷 Custom image: {args.image}")
    else:
        print(f"🏠 Scenario: {args.scenario}")
    
    print()
    
    # Run the test
    success = test_vamos_model(
        model_name_or_path=args.model,
        image_path=args.image,
        text_prompt=args.prompt,
        scenario=args.scenario,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        num_beams=args.num_beams,
        num_samples=args.num_samples,
        use_fp32=args.use_fp32
    )
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
