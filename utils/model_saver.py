import os
import torch


class ModelSaver:
    def __init__(self, objects_to_save, file_path):
        self.objects_dict = objects_to_save
        self.file_path = file_path

        s = file_path.split('/')
        if len(s) > 1:
            models_dir_path = '/'.join(s[:-1])

            if not os.path.exists(models_dir_path):
                os.makedirs(models_dir_path)

    def load(self, ignore_errors=False):
        try:
            if not os.path.exists(self.file_path):
                print('Model not found.')
                return

            print('Model found. Loading...')
            checkpoint = torch.load(self.file_path)

            for key, value in self.objects_dict.items():
                value.load_state_dict(checkpoint[key])
        except Exception as e:
            print(e)
            if not ignore_errors:
                raise e

    def save(self):
        torch.save({key: value.state_dict() for key, value in self.objects_dict.items()}, self.file_path)
