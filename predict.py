from experiment.train_test import train_test
from models.classifier import Classifier
from models.classifier import get_features
from utils.misc import load_model
from utils.extracted_dataloader import get_dloaders
from utils.eval import PrecisionRecallMetric
import hydra, os, logging, random, numpy as np, torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
#LAYERS = [f'feat_layer{i}/precompute_pca512' for i in range(1, 25)]
LAYERS = ['feat/precompute_pca512']
IN_DIM = 512

@hydra.main(config_path='./config', config_name='conf_extracted_test')
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
    classifier = Classifier(
        mode=cfg.mode, input_size=IN_DIM, n_layers=len(LAYERS),
    )
    classifier.load_state_dict(torch.load(cfg.ckpt_path)["classifier"])

    classifier = classifier.to(device)

    trainloader, valloader, testloader = get_dloaders(
        cfg=cfg, layers=LAYERS, logger=logger, g=g, is_training=False)
    logger.info("Dataset is {}".format(cfg.data))
 
    classifier.eval()
    all_preds = []
    all_labels = []

    dataloaders = {
        "train": trainloader,
        "valid": valloader,
        "test": testloader,    
    }

    for (split, dataloader) in sorted(dataloaders.items()):
        print(split)
        metric_tracker_harsh = PrecisionRecallMetric(tolerance=cfg.label_dist_threshold, mode="harsh")
        metric_tracker_lenient = PrecisionRecallMetric(tolerance=cfg.label_dist_threshold, mode="lenient")
        metric_tracker_label_harsh = PrecisionRecallMetric(tolerance=cfg.label_dist_threshold, mode="harsh")
        metric_tracker_label_lenient = PrecisionRecallMetric(tolerance=cfg.label_dist_threshold, mode="lenient")
        neg_metric_tracker_label_harsh = PrecisionRecallMetric(tolerance=0, mode="harsh")

        sigmoid = torch.nn.Sigmoid()
        logger.info(f"Extract segmentations for {len(dataloader.dataset)} samples")

        os.makedirs(cfg.out_path, exist_ok=True)
        with open(f'{cfg.out_path}/{split}_margin{int(cfg.margin*100)}.src', 'w') as f_src:
            progress = tqdm(ncols=80, total=len(dataloader))
            for samp in dataloader:
                features, segs, labels, _, lengths, fnames = samp
                segs = [[*segs[i][0]] + [s[1] for s in segs[i][1:]] for i in range(len(segs))]
                features = features.to(device).permute(2, 0, 1, 3)
                labels = labels.to(device)
                label_preds = [
                    torch.where(labels[i, :lengths[i]] == 1)[0].tolist() for i in range(labels.size(0))
                ]

                preds = classifier(features).squeeze()
                pred_scores = sigmoid(preds)
                preds = pred_scores > 0.5
                preds = [
                    torch.where(preds[i, :lengths[i]] == 1)[0].tolist() for i in range(preds.size(0))
                ]
                
                for i, pred in enumerate(preds):
                    clus_units = torch.zeros(lengths[i]).long() 
                    clus_units[pred] = 1
                    clus_units = clus_units.cumsum(dim=0)
                    if cfg.margin > 0:
                        score = pred_scores[i, :lengths[i]].squeeze(-1)
                        skip = (score >= 0.5-cfg.margin) * (score <= 0.5+cfg.margin)
                        clus_units[skip] = -1
                    clus_units = list(map(str, clus_units.tolist()))
                    print(' '.join([fnames[i]]+clus_units), file=f_src)
                   
                # Keep only predictions that the model is confident about
                if cfg.margin > 0:
                    preds = pred_scores > 0.5+cfg.margin
                    preds = [
                        torch.where(preds[i, :lengths[i]] == 1)[0].tolist() for i in range(preds.size(0))
                    ]
                    neg_preds = pred_scores < 0.5-cfg.margin
                    neg_preds = [
                        torch.where(neg_preds[i, :lengths[i]] == 1)[0].tolist() for i in range(neg_preds.size(0))
                    ]
                    neg_segs = [[t for t in range(lengths[i]) if not t in segs[i]] for i in range(len(segs))]
                    neg_metric_tracker_label_harsh.update(neg_segs, neg_preds)

                # for seg, pred in zip(segs, preds):
                #     seg = np.array(seg, dtype=np.int)
                #     pred = np.array(pred, dtype=np.int)
                #     print("GT: ", seg)
                #     print("Pred: ", pred)
                #     print("match_counts, dup_counts: ", metric_tracker_label_harsh.get_counts(seg, pred))

                metric_tracker_harsh.update(segs, preds)
                metric_tracker_lenient.update(segs, preds)
                metric_tracker_label_harsh.update(segs, label_preds)
                metric_tracker_label_lenient.update(segs, label_preds)
                progress.update(1)
            progress.close()
        logger.info("Computing metrics with distance threshold of {} frames".format(cfg.label_dist_threshold))

        tracker_metrics_harsh = metric_tracker_harsh.get_stats()
        tracker_metrics_lenient = metric_tracker_lenient.get_stats()
        tracker_metrics_label_harsh = metric_tracker_label_harsh.get_stats()
        tracker_metrics_label_lenient = metric_tracker_label_lenient.get_stats()
        if cfg.margin > 0:
            tracker_neg_metrics_label_harsh = neg_metric_tracker_label_harsh.get_stats()
            for k in tracker_neg_metrics_label_harsh.keys():
                logger.info(
                    "{:<15} {:>10.4f} {:>10.4f}".format(k+" for negative labels:",
                    tracker_neg_metrics_label_harsh[k],
                    tracker_neg_metrics_label_harsh[k])
                )

        logger.info(f"{'SCORES:':<15} {'Lenient':>10} {'Harsh':>10}")
        for k in tracker_metrics_harsh.keys():
            logger.info("{:<15} {:>10.4f} {:>10.4f}".format(k+" for pseudo labels:", tracker_metrics_label_lenient[k], tracker_metrics_label_harsh[k]))

        for k in tracker_metrics_harsh.keys():
            logger.info("{:<15} {:>10.4f} {:>10.4f}".format(k+":", tracker_metrics_lenient[k], tracker_metrics_harsh[k]))

if __name__ == '__main__':
    main()
