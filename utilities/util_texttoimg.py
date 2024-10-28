import requests
import os
import io
from PIL import Image

# Set up the API URL
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

def query(payload, api_key):
    """Send a POST request to Hugging Face's API and return the generated image bytes."""
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"Sending request to Hugging Face API with prompt: {payload['inputs']}")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad requests (4XX, 5XX)
        print("Received response from Hugging Face API.")
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        print(f"Response content: {response.content}")
        return None

def generate_images_from_prompts(post_data, folder_name="images", api_key=None):
    """Generate images from text prompts via Hugging Face API and save them to a specified folder."""
    if api_key is None:
        raise ValueError("API key must be provided.")
    
    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Iterate through each post and its prompts
    for post_id, content in post_data.items():
        prompts = content.get('Prompt', [])
        print("prompts:", prompts)
        
        for i, prompt in enumerate(prompts):
            # Print out the prompt being processed
            print(f"Generating image for {post_id}, Prompt {i+1}: {prompt}")
            
            # Query the Hugging Face API to generate the image
            image_bytes = query({"inputs": prompt}, api_key)
            
            if image_bytes is None:
                print(f"Failed to generate image for {post_id}, Prompt {i+1}.")
                continue  # Skip to the next prompt
            
            try:
                # Load the image from the returned bytes
                image = Image.open(io.BytesIO(image_bytes))
                
                # Define the file path for saving the image
                image_filename = os.path.join(folder_name, f"{post_id}_prompt_{i+1}.png")
                
                # Save the image to the specified path
                image.save(image_filename)
                print(f"Saved image {post_id}_prompt_{i+1} to {image_filename}")
            except Exception as e:
                print(f"Failed to process image for {post_id}, Prompt {i+1}: {e}")