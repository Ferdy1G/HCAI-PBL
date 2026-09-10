
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache

import json
from palmerpenguins import load_penguins

from .services.trainer import ModelTrainer, generate_counterfactuals_model, generate_pdp_plot


def index(request):
    """Renders the main page view."""
    return render(request, "project2/index.html", {
        "selected_model": "dt",
        "lambda_val": "0.002",
    })

def train_model(request):
    if request.method == "POST":
        selected_model = request.POST.get("model", "dt")
        lambda_val = request.POST.get("lambda", "0.05")

        trainer = ModelTrainer(model_type=selected_model, lambda_val=lambda_val)
        results = trainer.train_and_evaluate()

        # 1. Ensure a session exists BEFORE querying session_key
        if not request.session.session_key:
            request.session.create()

        session_key = request.session.session_key

        # 2. Save active pipeline model using clean string key
        cache.set(
            f"model_{session_key}", 
            {"model": trainer.best_pipeline}, 
            3600
        )

        return render(request, "project2/index.html", {
            "selected_model": selected_model,
            "lambda_val": lambda_val,
            "results": results,
        })

def get_sample(request):
    """Retrieves single raw penguin record by index ID dynamically."""
    sample_id = request.GET.get("sample_id")
    try:
        df = load_penguins().dropna().reset_index(drop=True)
        idx = int(sample_id)
        if 0 <= idx < len(df):
            row = df.iloc[idx].to_dict()
            return JsonResponse({"sample": row})
    except (ValueError, TypeError):
        pass
        
    return JsonResponse({"sample": None})

def generate_counterfactuals(request):
    """View endpoint that receives fetch parameters from Alpine.js."""
    if request.method == "POST":
        data = json.loads(request.body)
        
        result = generate_counterfactuals_model(
            session_key=request.session.session_key,
            sample_id=data.get("sample_id"),
            num_samples=data.get("num_samples", 50),
            target_class=data.get("target_class")
        )
        return JsonResponse({"counterfactual": result})

    return JsonResponse({"error": "Invalid request"}, status=400)

def get_pdp_plot(request):
    col = request.GET.get('col', 'bill_length_mm')
    session_key = request.session.session_key
    
    pdp_url = generate_pdp_plot(session_key, col)

    return JsonResponse({"plot_url": pdp_url})