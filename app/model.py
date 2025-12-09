from transformers import RobertaForSequenceClassification

def get_model(num_labels: int = 8):
    """
    Factory function to initialize the RoBERTa model architecture.
    """
    return RobertaForSequenceClassification.from_pretrained(
        'FacebookAI/roberta-base',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
