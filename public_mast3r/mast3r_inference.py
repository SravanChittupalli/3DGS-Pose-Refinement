import numpy as np
import torch
from matplotlib import pyplot as pl
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from PIL import Image

class ImageMatcher:
    def __init__(self, model_path, device='cuda', size=512, lr=0.01, niter=300):
        self.device = device
        self.size = size
        self.lr = lr
        self.niter = niter
        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(device)

    # Helper function for resizing a PIL image
    def _resize_pil_image(self, image, size):
        """Helper function to resize a PIL image while maintaining aspect ratio."""
        return image.resize((size, size), Image.LANCZOS)

    # Mock ImgNorm function for normalization
    def ImgNorm(self, img):
        """Normalize the image (dummy function; replace with actual implementation)."""
        return np.asarray(img) / 255.0  # Assuming normalization between 0 and 1

    def load_image(self, np_image, square_ok=False, verbose=False):
        """ Open and convert a single NumPy RGB image to proper input format for DUSt3R.
        
        Args:
        - np_image (np.ndarray): A NumPy array representing an RGB image with shape (H, W, C).
        - size (int): The size for resizing.
        - square_ok (bool): Whether the input can be kept square (specific to cropping logic).
        - verbose (bool): Whether to print details about the image.

        Returns:
        - A dictionary containing the image tensor and metadata for DUSt3R.
        """
        # Check if input is a NumPy array and has correct dimensions
        if not isinstance(np_image, np.ndarray):
            raise ValueError(f'Expected an np.ndarray, got {type(np_image)}')
        if np_image.ndim != 3 or np_image.shape[2] != 3:
            raise ValueError(f'Expected image with shape (H, W, 3), got {np_image.shape}')

        # Convert NumPy array to PIL Image
        img = Image.fromarray(np_image)

        # W1, H1 = img.size
        # if self.size == 224:
        #     # Resize short side to 224 (then crop)
        #     img = self._resize_pil_image(img, round(self.size * max(W1 / H1, H1 / W1)))
        # else:
        #     # Resize long side to 512
        #     img = self._resize_pil_image(img, self.size)

        # W, H = img.size
        # cx, cy = W // 2, H // 2

        # if self.size == 224:
        #     half = min(cx, cy)
        #     img = img.crop((cx - half, cy - half, cx + half, cy + half))
        # else:
        #     halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        #     if not square_ok and W == H:
        #         halfh = 3 * halfw // 4
        #     img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        # W2, H2 = img.size

        # if verbose:
        #     print(f' - Loaded image with resolution {W1}x{H1} --> {W2}x{H2}')

        # Normalize the image (assuming ImgNorm is a normalization function you have)
        img_tensor = self.ImgNorm(img)

        # Add batch dimension and convert to PyTorch tensor
        img_tensor = torch.from_numpy(np.array(img_tensor)).permute(2, 0, 1).float()[None]

        # Create the image information dictionary
        img_info = dict(
            img=img_tensor,  # The final tensor (BxCxHxW)
            true_shape=np.int32([img.size[::-1]]),  # Original image size as (H, W)
            idx=0,
            instance='0'
        )

        return img_info
    
    def infer_and_match_tensors(self, image_tensor1, image_tensor2, true_shape1, true_shape2, visualize=None):
        """
        Takes two image tensors as input and performs inference and matching.
        :param image_tensor1: Tensor for the first image (1xCxHxW).
        :param image_tensor2: Tensor for the second image (1xCxHxW).
        :param true_shape1: Shape of the first image before resizing (HxW).
        :param true_shape2: Shape of the second image before resizing (HxW).
        :param n_viz: Number of matches to visualize.
        """
        images = [
            dict(img=image_tensor1, true_shape=true_shape1, idx=0, instance='0'),
            dict(img=image_tensor2, true_shape=true_shape2, idx=1, instance='1'),
        ]

        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=self.device, dist='dot', block_size=2**13
        )

        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        if visualize:
            self.visualize_matches(view1, view2, matches_im0, matches_im1, 30, H0, W0, H1, W1, save_path=visualize)

        return matches_im0, matches_im1

    def infer_and_match(self, image_path1, image_path2, n_viz=20):
        images = load_images([image_path1, image_path2], size=self.size)
        '''
        images[0].keys()
        dict_keys(['img', 'true_shape', 'idx', 'instance'])

        images[0]['img'].shape
        torch.Size([1, 3, 384, 512])
        '''
        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=self.device, dist='dot', block_size=2**13
        )

        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        self.visualize_matches(view1, view2, matches_im0, matches_im1, n_viz, H0, W0, H1, W1)

    def visualize_matches(self, view1, view2, matches_im0, matches_im1, n_viz, H0, W0, H1, W1, save_path="matcning.png"):
        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)

        pl.figure()
        pl.imshow(img)
        # pl.savefig("test_1.png")
        cmap = pl.get_cmap('jet')

        match_idx_to_viz = np.round(np.linspace(0, matches_im0.shape[0] - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)
        pl.savefig(save_path)

# Example Usage
if __name__ == '__main__':
    model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    matcher = ImageMatcher(model_path)
    # matcher.infer_and_match(
    #     '../public_marepo/datasets/7scenes_redkitchen/test/rgb/seq-03-frame-000276.color.png',
    #     '../public_marepo/datasets/7scenes_redkitchen/test/rgb/seq-03-frame-000250.color.png'
    # )
    matcher.infer_and_match(
        '../public_marepo/datasets/7scenes_chess/train/rgb/seq-01-frame-000276.color.png',
        '../public_marepo/datasets/7scenes_chess/train/rgb/seq-01-frame-000250.color.png'
    )