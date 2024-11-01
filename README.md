# Auto-content

## How to use
Can only be used locally by running the following command 'python main.py', which will start the pipeline and outputs a short-form video for SoMe

## Pipeline
URL -> Research -> Content -> Text-to-speech -> Text-to-image -> Video production
1. Fetches URLs for football news which are passed forward to the research chain
   - Scrapes for football articles using News API
   - Scrapes news from yesterday
   - Uses football-related keywords to exclude non-football articles
   - Clusters news to avoid similar news stories
   - Creates embeddings from articles' text using distilbert-base-uncased
   - Stores article embeddings in MongoDB vector database with metadata

2. Run research chain:
   - Agents: research agent, list agent, and supervisor (not used)
   - Graph flow: research agent -> list agent -> END
   - Research agent: Scrapes the URLs using JINA's reader API
   - List agent: Creates lists of news [news story 1, news story 2, etc.]

3. Content chain:
   - Agents: text agent, description agent, prompt generator, dictionary generator, supervisor (not used)
   - Graph flow: text agent -> description agent -> prompt agent -> dictionary agent -> END
   - Text agent: Generates the text that will be used as audio in the video using tts
   - Description agent: Generates the text for the post
   - Prompt agent: Creates the text-to-image prompt for image generation
   - Dictionary agent: Uses the outputs from above to generate a dictionary with related information for each post using the following structure

4. Text-to-speech (TTS)
   - Translates the text from the post's dictionary to audio using 'parler-tts/parler-tts-mini-expresso' model from huggingface

5. Text-to-image (TTI)
   - Creates images from the prompt key's value in the the post's dictionary, which is done using FLUX schnell's API available on huggingface
  
6. Video generation
   - Combines generated audio and images into a video that is saved locally


## Todo
- Use LLM agent to determine if the news is about football instead of keywords
- Use different logic than n-clusters=2 for clustering articles
- Containerization
- Scheduling
- Use State to store agent outputs instead of returning the raw string and transforming to the right format
- Better tts model finetuned for football player names
- Better tti model finetuned for football players - depends on the purpose
- Include closed caption in video generation
