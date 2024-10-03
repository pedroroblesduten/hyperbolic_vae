from trainers.trainer_vae import TrainerVAE
from config import get_args

def main():
    config = get_args()

    TrainerVAE(**config.__dict__).run()

if __name__ == "__main__":
    main()