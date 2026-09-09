# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Ferdinand Grenzing", "matriculation": "501309"},
    ]
    
    projects = [
        {"name": "demos", "url_name": "demos:index"},
        {"name": "Project 1", "url_name": "project1:workspace_view"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))
