import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, gt_keypoints=None):
        """Directly extract features from the backbone+neck."""
        # yaxian add gt_keypoints
        if gt_keypoints == None:
            x = self.backbone(img)
        else:
            x = self.backbone(img, gt_keypoints)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img, gt_keypoints=None):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # yaxian add gt_keypoints
        x = self.extract_feat(img, gt_keypoints)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_keypoints (list[Tensor]): Each item are the truth keypoints for each
                image in [x,y,v, ..., x,y,v] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert not (gt_keypoints == None)
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # yaxian add gt_keypoints
        for i in range(gt_keypoints.size()[0]):
            x_factor = img_metas[i]['scale_factor'][0]
            y_factor = img_metas[i]['scale_factor'][1]
            for j in range( int(gt_keypoints.size()[1] / 3)):
                if 'border' in img_metas[i].keys():
                    gt_keypoints[i][j*3] = int(x_factor*(gt_keypoints[i][j*3] + img_metas[i]['border'][2])) +1 
                    gt_keypoints[i][j*3 + 1] = int(y_factor*(gt_keypoints[i][j*3 + 1] + img_metas[i]['border'][0])) +1
                else:
                    gt_keypoints[i][j*3] = int(x_factor*gt_keypoints[i][j*3])
                    gt_keypoints[i][j*3 + 1] = int(y_factor*gt_keypoints[i][j*3 + 1])

        x = self.extract_feat(img, gt_keypoints)
        #print('self.backbone.patch.size()', self.backbone.patch.size())
        #print('gt_labels', gt_labels)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,gt_labels,
                        patch=self.backbone.patch, gt_bboxes_ignore=gt_bboxes_ignore)
        #yaxian
        '''
        for name, parms in self.backbone.named_parameters():
            if (parms.requires_grad):
                print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		            ' -->grad_value:',parms.grad)
        
        for name, parms in self.bbox_head.named_parameters():	
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		    ' -->grad_value:',parms.grad)
        '''
        return losses

    '''
    def forward_test(self, img, img_metas, gt_keypoints=None, rescale=False):
        # yaxian add keypoints here but not use it
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # add keypoints,yaxian
        print('type(img)', type(img))
        print('len(img)', len(img))
        print('type(img[0])', type(img[0]))
        print('img[0].size()', img[0].size())
        #print('img[0]', img[0])

        if gt_keypoints == None:
            img = img[0]
            img_metas = img_metas[0]
            x = self.extract_feat(img)
        else:
            img = torch.tensor(img[0])#.squeeze()
            gt_keypoints = torch.tensor(gt_keypoints[0])#.squeeze()
            img_metas = img_metas[0]
            #print('img.size()', img.size())
            #print('gt_keypoints.size()', gt_keypoints.size())
            #print('type(img_metas)', type(img_metas))
            x = self.extract_feat(img, gt_keypoints)

        

        outs = self.bbox_head(x)
        #print('img_metas', img_metas)
        #img_metas = img_metas.data[0] #yaxian
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale, with_nms=False)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        print('x[0].size()', x[0].size())
        print('len(x)', len(x))
        print('len(outs)', len(outs))
        print('len(outs[0])', len(outs[0]))
        print('len(outs[0][0])', len(outs[0][0]))
        print('len(outs[0][0][0])', len(outs[0][0][0]))
        print('len(bbox_list)', len(bbox_list))
        print('type(bbox_list)', type(bbox_list))

        #print('bbox_list[0]', bbox_list[0])
        print('bbox_list[0][0].size()', bbox_list[0][0].size()) #[n,4] ? why 4 not 5
        print('bbox_list[0][1].size()', bbox_list[0][1].size()) #[n,]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list #yaxian
        ]
        return bbox_results
    '''
    
    def simple_test(self, img, img_metas, gt_keypoints=None, rescale=False):
        # already exist function, add keypoints here
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        if gt_keypoints==None:
            x = self.extract_feat(img)
        else:
            
            gt_keypoints = torch.tensor(gt_keypoints[0])
            x = self.extract_feat(img, gt_keypoints)

        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
