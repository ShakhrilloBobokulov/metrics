# Generated by Django 5.1 on 2024-08-30 11:08

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('visit', '0009_forecastresults'),
    ]

    operations = [
        migrations.AddField(
            model_name='forecastresults',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='forecastresults',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
