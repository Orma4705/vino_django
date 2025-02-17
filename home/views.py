from django.shortcuts import render
from wines.models import Wine
from diffusers import StableDiffusionXLPipeline
import torch
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")

def index(request):
    wines = Wine.objects.all().order_by("-id")[:6]
    prompt = """{wines.description}. Cold color palette, muted colors, 8k."""
    image = pipe(prompt, num_inference_steps=50, output_type="pil").images[0]
    return render(request, "home/index.html", {"wines": wines},{"image": image})


def about(request):
    return render(request, "home/about.html")


def services(request):
    return render(request, "home/services.html")


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
