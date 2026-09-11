from django.shortcuts import render


def index(request):
    """Renders the landing page for Project 3 with the report download button."""
    return render(
        request,
        "project3/index.html",
        {"pdf_url": "https://drive.google.com/file/d/10Ul0xPsYiyhxZM-dSXkygv2X1Cnr8__g/view?usp=drive_link"},
    )