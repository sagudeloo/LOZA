"""LOZA URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from LOZA import views
from LOZA.views import trazlin,viewTrazlin,search,viewSearch,BisecReg,viewBisecReg,NewtPoint,viewNewtPoint,RaizMul,viewRaizMul

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('incrementalS/', views.incrementalSearch),
    path('lufact/', views.LUFact),
    path('gauss/', views.gauss)
    path('trazlin/', trazlin),
    path('viewTrazlin/', viewTrazlin),
    path('Search/', search),
    path('viewSearch/', viewSearch),
    path('BisecReg/', BisecReg),
    path('viewBisecReg/', viewBisecReg),
    path('NewtPoint/', NewtPoint),
    path('viewNewtPoint/', viewNewtPoint),
    path('RaizMul/', RaizMul),
    path('viewRaizMul/', viewRaizMul),
]
