# Generated by Django 5.1.6 on 2025-02-12 17:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wines', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='vino',
            name='imagen',
            field=models.ImageField(default='ruta/a/una/imagen/predeterminada.jpg', upload_to='vinos/'),
        ),
    ]
