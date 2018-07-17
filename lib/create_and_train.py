from create_models import main as create
from train import train

if __name__ == '__main__':
    identifier = create()
    train(identifier)
