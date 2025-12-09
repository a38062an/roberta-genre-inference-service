import torch
import time
import os
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from app.config import MODEL_PATH, MAX_LEN, DEVICE, GENRE_COLUMNS, THRESHOLDS

class InferenceService:
    def __init__(self):
        # Load model at startup to avoid latency on individual requests
        print(f"Loading model from {MODEL_PATH}...")
        self.tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')
        
        # Initialize model architecture
        self.model = RobertaForSequenceClassification.from_pretrained(
            'FacebookAI/roberta-base',
            num_labels=8,
            problem_type="multi_label_classification"
        )
        
        # Load your trained weights if they exist
        if os.path.exists(MODEL_PATH):
            # Load to CPU first to avoid MPS unaligned blit errors
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.to(DEVICE)
            print("Model weights loaded from artifacts.")
        else:
            print(f"WARNING: Model weights not found at {MODEL_PATH}. Using random initialization for testing.")
            
        self.model.to(DEVICE)
        self.model.eval()
        print("Model loaded successfully.")

    def predict(self, text: str):
        start_time = time.time()
        
        # Tokenize input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Sigmoid activation
            probs = torch.sigmoid(outputs.logits).squeeze()

        # Apply thresholds to determine genres
        # Convert tensor to python dict
        results = {}
        for idx, genre in enumerate(GENRE_COLUMNS):
            # Compare prob against specific threshold for that genre
            threshold = THRESHOLDS.get(genre, 0.5) 
            # Check if probs is a scalar or 1-d tensor
            score = probs[idx].item() if probs.ndim > 0 else probs.item()
            results[genre] = 1 if score >= threshold else 0

        exec_time = (time.time() - start_time) * 1000
        return results, exec_time

# Global instance
inference_service = InferenceService()
