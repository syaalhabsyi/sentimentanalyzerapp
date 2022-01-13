from django.apps import AppConfig
from django.conf import settings
import os
import pickle


class AisentimentanalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'aisentimentanalyzer'
    path = os.path.join(settings.MODELS, 'models.p')

    # load models into separate variables
    # these will be accessible via this class
    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)
    model = data['model']
    vectorizer = data['vectorizer']