import json
import os
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from .services.preference_engine import get_random_movies, update_plackett_luce
import time

df_movies = pd.read_csv(settings.CSV_PATH) if os.path.exists(settings.CSV_PATH) else pd.DataFrame()

def landing(request):
    """Landing page: handles cohort assignment and sets up task timers."""
    if request.method == "POST":
        cohort = request.POST.get("cohort", "A")
        request.session["cohort"] = cohort
        request.session["step"] = 1
        request.session["start_time"] = time.time()  # Track task start time
        request.session["weights"] = [0.0] * 50

        first_design = "design1" if cohort == "A" else "design2"
        return redirect(f"project4:{first_design}")

    return render(
        request,
        "project4/landing.html",
        {"pdf_url": "https://drive.google.com/your-pdf-link-here"},
    )

def design1_pairwise(request):
    """Design 1 UI: Pairwise comparison."""
    return render(request, "project4/design1_pairwise.html")

def design2_ranking(request):
    """Design 2 UI: 10-item ranking."""
    return render(request, "project4/design2_ranking.html")

def get_candidates(request):
    """API: Fetch candidate movies (2 for Design 1, 10 for Design 2)."""
    count = int(request.GET.get("count", 2))
    movies = get_random_movies(df_movies, n=count)
    return JsonResponse({"movies": movies})

def submit_preference(request):
    """API: Receives preference updates and auto-advances if 90s have elapsed."""
    if request.method == "POST":
        # 1. Fallback initialization if start_time was missing
        if "start_time" not in request.session:
            request.session["start_time"] = time.time()

        start_time = request.session["start_time"]
        elapsed = time.time() - start_time
        time_limit = 90  # seconds

        # Debug print in server terminal to verify timer
        print(f"[DEBUG] Elapsed: {elapsed:.1f}s / {time_limit}s")

        next_url = None

        if elapsed >= time_limit:
            step = request.session.get("step", 1)
            cohort = request.session.get("cohort", "A")

            if step == 1:
                # Transition to Step 2 & reset timer for the second task
                request.session["step"] = 2
                request.session["start_time"] = time.time()
                next_url = (
                    "/project4/design2/"
                    if cohort == "A"
                    else "/project4/design1/"
                )
            else:
                # Both tasks complete
                next_url = "/project4/complete/"

        # Force Django to commit session changes immediately
        request.session.modified = True

        return JsonResponse(
            {
                "status": "success",
                "elapsed": round(elapsed, 1),
                "next_url": next_url,
            }
        )

    return JsonResponse({"error": "Invalid method"}, status=400)

def complete(request):
    """Completion screen after study."""
    return render(request, "project4/complete.html")