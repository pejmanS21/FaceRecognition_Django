from attr import field
from matplotlib.pyplot import cla
from .models import Post
from django import forms


class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['image']