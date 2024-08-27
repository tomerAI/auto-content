import os
from moviepy.editor import ImageSequenceClip, AudioFileClip
from moviepy.editor import VideoFileClip
from moviepy.editor import TextClip, CompositeVideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import ImageSequenceClip, AudioFileClip, TextClip, CompositeVideoClip
import os

from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def add_closed_captions(clip, post_data, caption_duration=2):
    """
    Adds closed captions to a video clip using Pillow to render text as images.
    
    Parameters:
    - clip: The video clip to add captions to.
    - post_data: The data for the post, containing the text to caption.
    - caption_duration: Duration (in seconds) for each caption segment (default is 2 seconds).
    
    Returns:
    - A new video clip with captions added.
    """
    # Get the text from the post_data
    text = post_data.get('Text', "")
    
    # Split the text into smaller segments (5-6 words per caption)
    words = text.split()
    captions = []
    words_per_caption = 6  # Number of words per caption

    # Break the text into chunks of 6 words
    for i in range(0, len(words), words_per_caption):
        captions.append(" ".join(words[i:i + words_per_caption]))
    
    # Get the width and height of the video
    video_width, video_height = clip.size
    
    # Font settings (change the font path to a valid TTF file if needed)
    font = ImageFont.load_default()
    
    # Create caption images and overlay them
    def make_frame(t):
        # Get the current frame
        frame = clip.get_frame(t)
        pil_frame = Image.fromarray(frame)

        # Determine which caption should be shown at time t
        caption_index = int(t // caption_duration)
        if caption_index < len(captions):
            draw = ImageDraw.Draw(pil_frame)
            caption_text = captions[caption_index]
            
            # Measure text size
            text_width, text_height = draw.textsize(caption_text, font=font)
            
            # Calculate text position (centered at the bottom)
            text_x = (video_width - text_width) // 2
            text_y = video_height - text_height - 30  # 30 pixels above the bottom edge
            
            # Draw a black rectangle as background
            draw.rectangle(
                [(text_x - 10, text_y - 10), (text_x + text_width + 10, text_y + text_height + 10)], 
                fill="black"
            )
            
            # Draw the caption text
            draw.text((text_x, text_y), caption_text, font=font, fill="white")
        
        return np.array(pil_frame)

    # Apply the function to each frame of the video
    final_clip = clip.fl_image(make_frame)
    
    return final_clip


def create_videos_from_posts(posts_dict, fps=1, output_folder='output'):
    """
    Creates videos by combining images and audio for each post based on the posts_dict structure.
    
    Parameters:
    - posts_dict: Dictionary containing information for each post (audio, images, captions).
    - fps: Frames per second for images.
    - output_folder: The directory to save the output videos.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over each post in the dictionary
    for post_id, post_data in posts_dict.items():
        print(f"Processing {post_id}...")

        # Load the audio file for the post
        audio_file = os.path.join('audio', f'{post_id}_audio.wav')
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file for {post_id} not found at {audio_file}.")
        
        audio_clip = AudioFileClip(audio_file)
        
        # Load the images for the post
        image_files = []
        image_folder = 'images'
        for file in sorted(os.listdir(image_folder)):
            if file.startswith(f'{post_id}_prompt_') and file.endswith('.png'):
                image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found for {post_id}.")
        
        # Create an image sequence clip from the list of image files
        clip = ImageSequenceClip(image_files, fps=fps)
        
        final_output_folder = 'output_final'
        # Set the audio for the video
        final_clip = clip.set_audio(audio_clip)

        # Call the external function to add closed captions
        final_clip_with_captions = add_closed_captions(final_clip, post_data)
        
        # Define the output path for the final video with captions
        output_video_with_captions_path = os.path.join(final_output_folder, f'{post_id}_video_with_captions.mp4')
        
        # Write the final video to file with captions
        final_clip_with_captions.write_videofile(output_video_with_captions_path, codec='libx264', audio_codec='aac')
        
        print(f"Video for {post_id} created with captions: {output_video_with_captions_path}")
        
       


# Ensure compatibility with newer Pillow versions
if hasattr(Image, 'Resampling'):
    RESAMPLING_METHOD = Image.Resampling.LANCZOS
else:
    RESAMPLING_METHOD = Image.ANTIALIAS

def resize_frame(frame, new_size):
    """
    Resizes a single video frame using Pillow's updated resampling method.
    
    Parameters:
    - frame: The frame to be resized (as a numpy array).
    - new_size: The target size (width, height).
    
    Returns:
    - The resized frame.
    """
    pil_image = Image.fromarray(frame)
    resized_image = pil_image.resize(new_size, RESAMPLING_METHOD)
    return np.array(resized_image)

def postprocess_videos(input_folder='output', output_folder='processed_videos', tiktok_resolution=(1080, 1920)):
    """
    Post-processes the generated videos to make them compatible with TikTok's format.
    
    Parameters:
    - input_folder: The folder where the original videos are located.
    - output_folder: The folder where the TikTok-formatted videos will be saved.
    - tiktok_resolution: The resolution for TikTok videos (default is 1080x1920 for 9:16 aspect ratio).
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over each video in the input folder
    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4'):
            input_video_path = os.path.join(input_folder, video_file)
            output_video_path = os.path.join(output_folder, video_file)
            
            # Load the video
            video_clip = VideoFileClip(input_video_path)
            
            # Apply the resize to each frame
            video_resized = video_clip.fl_image(lambda frame: resize_frame(frame, tiktok_resolution))
            
            # Write the final video in the correct TikTok format
            video_resized.write_videofile(output_video_path, codec='libx264', audio_codec='aac', 
                                          fps=24, preset="fast", threads=4)
            
            print(f"Processed TikTok video saved as: {output_video_path}")