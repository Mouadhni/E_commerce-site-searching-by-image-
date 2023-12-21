from django import  forms

from myapp.models import Product


class Productform(forms.ModelForm):
      class Meta:
          model=Product
          fields ='__all__'



