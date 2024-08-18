import spacy

class PreprocessingModule:
    def __init__(self):
        # Ensure the model is downloaded
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
