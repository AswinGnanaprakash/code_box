# Generated by Django 2.0.10 on 2020-02-25 19:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='plant',
            old_name='Main_Img',
            new_name='upload_your_image',
        ),
    ]
