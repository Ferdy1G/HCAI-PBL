import uuid
from django.db import models

class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_file = models.FileField(upload_to='datasets/originals/')
    working_file_path = models.CharField(max_length=500, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Dataset {self.id}"


class TrainedModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='models')
    
    # Configuration
    algorithm = models.CharField(max_length=50)       # e.g., 'Decision Tree'
    target_column = models.CharField(max_length=100)   # e.g., 'Target'
    hyperparameters = models.JSONField(default=dict)    # e.g., {"max_depth": 5, "val_split": 0.15}
    
    # Performance Metrics
    metrics = models.JSONField(default=dict)            # e.g., {"train_accuracy": 0.88, "val_accuracy": 0.82}
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.algorithm} ({self.target_column}) - {self.created_at.strftime('%H:%M:%S')}"