import matplotlib.pyplot as plt
from pathlib import Path

from recon import ReCoN
from recon_cifar import CIFARReCoNBuilder
from train_common import get_loaders, load_model_from_pt
from recon_visualizer_utils import visualize_recon

if __name__ == "__main__":

    _, test_loader = get_loaders("./data", 1)
    
    recon_builder = CIFARReCoNBuilder(ReCoN(1))
    vision_model, device = load_model_from_pt(Path(__file__).resolve().parent / 'best_resnet18_cifar100_super.pt')

    for idx, (image, _ ) in enumerate(test_loader):
        image = image.to(device)
        
        previous_net_num = recon_builder.net.num
        recon_builder.build(image, vision_model)
        new_node_indices = range(previous_net_num, recon_builder.net.num)
        print(f"Processed image {idx}, added {len(new_node_indices)} nodes.")

        visualize_recon(recon_builder, title="ReCoN CIFAR 100", save_png=None, sample_image=image[0], new_node_indices=new_node_indices)
        plt.show()
        
 