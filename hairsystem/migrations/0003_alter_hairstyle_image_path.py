# Generated by Django 5.0.4 on 2024-05-18 07:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hairsystem', '0002_hairstyle_image_path'),
    ]

    operations = [
        migrations.AlterField(
            model_name='hairstyle',
            name='image_path',
            field=models.ImageField(default='default.jpg', upload_to='uploads/'),
        ),
    ]