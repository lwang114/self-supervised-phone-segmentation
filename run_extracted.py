from experiment.train_test_extracted import train_test
from models.classifier import Classifier
from utils.misc import load_model
from utils.extracted_dataloader import get_dloaders
import hydra, os, logging, random, numpy as np, torch
#LAYERS = [f'feat_layer{i}/precompute_pca512' for i in range(1, 25)]
LAYERS = ['feat/precompute_pca512']
IN_DIM = 512
#LAYERS = [f'feat_layer{i}' for i in range(7, 19)]
#IN_DIM=1024

@hydra.main(config_path='./config', config_name='conf_extracted')
def main(cfg):

    logger = logging.getLogger(__name__)

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda')

    logger.info("Model base checkpoint is {}".format(cfg.base_ckpt_path))

    logger.info("Instantiating model...")
    classifier = Classifier(mode=cfg.mode, input_size=IN_DIM, n_layers=len(LAYERS))
    classifier = classifier.to(device)

    trainloader, valloader, testloader = get_dloaders(
        cfg=cfg, logger=logger, layers=LAYERS, g=g)
    logger.info("Dataset is {}".format(cfg.data))

    train_test(cfg, classifier, trainloader, valloader, testloader, logger)

if __name__ == "__main__":
    main()
