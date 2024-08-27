import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os

def text_to_speech_conversion(post_data, key="Text", folder_name="audio"):
    """Convert the 'Text' from the dictionary into audio using the Parler-TTS model with a sports commentator style voice."""

    # Check if CUDA is available, otherwise fallback to CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

    # Set a default description for a sports commentator voice
    sports_commentator_description = (
        "Thomas' voice is energetic, clear, and fast-paced, with a sense of excitement."
    )

    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for post_id, content in post_data.items():
        # Extract the text to be converted to speech
        prompt = content.get(key, "")

        if not prompt:
            print(f"Skipping {post_id} due to missing text.")
            continue

        # Print the text being converted (for verification)
        print(f"Converting {post_id}: {prompt}")

        # Tokenize the inputs (using the fixed sports commentator description)
        input_ids = tokenizer(sports_commentator_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate audio
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

        # Convert the generated audio tensor to numpy
        audio_arr = generation.cpu().numpy().squeeze()

        # Define the path for the audio file in the specified folder
        filename = os.path.join(folder_name, f"{post_id}_audio.wav")

        # Save the audio as a .wav file using soundfile
        sf.write(filename, audio_arr, model.config.sampling_rate)

        print(f"Saved {post_id} to {filename}")