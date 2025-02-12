from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render


def index(request):
    return render(request, "home/index.html")


def about(request):
    template = loader.get_template("home/about.html")
    return HttpResponse(template.render())


def services(request):
    template = loader.get_template("home/services.html")
    return HttpResponse(template.render())


def contact(request):
    if request.method == "POST":
        nombre = request.POST.get("nombre")
        email = request.POST.get("email")
        mensaje = request.POST.get("mensaje")
        return render(
            request,
            "home/success.html",
            {
                "nombre": nombre,
                "email": email,
                "mensaje": mensaje,
            },
        )

    return render(request, "home/contact.html")
