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
        {"name": "Project 2", "url_name": "project2:index"},
        {"name": "Project 3", "url_name": "project3:index"},
        {"name": "Project 4", "url_name": "project4:landing"},
    ]

    context = {
        "students": students,
        "projects": projects,
    }

    return HttpResponse(template.render(context, request))