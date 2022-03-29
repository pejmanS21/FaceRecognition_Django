from django.contrib import admin
from .models import Post


# Register your models here.
@admin.register(Post)
class PostImage(admin.ModelAdmin):
    list_display = ['image']
