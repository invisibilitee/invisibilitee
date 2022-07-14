# InvisibiliTee: Angle-agnostic Cloaking from Person-Tracking Systems with a Tee
This repo is an official implementation of `InvisibiliTee: Angle-agnostic Cloaking from Person-Tracking Systems with a Tee`, accepted by ICANN 2022. 



> By Yaxian Li, Bingqing Zhang, Guoping Zhao, Mingyu Zhang, Jiajun Liu, Ziwei Wang, and Jirong Wen

After a survey for person-tracking system-induced privacy concerns, we propose a black-box adversarial attack method on state- of-the-art human detection models called InvisibiliTee. The method learns printable adversarial patterns for T-shirts that cloak wearers in the physical world in front of person-tracking systems. We design an angle-agnostic learning scheme which utilizes segmentation of the fash- ion dataset and a geometric warping process so the adversarial patterns generated are effective in fooling person detectors from all camera angles and for unseen black-box detection models. Empirical results in both digital and physical environments show that with the InvisibiliTee on, person-tracking systems’ ability to detect the wearer drops significantly.

![framework](/Users/brucezhang/lab/mmdet-yx/resources/framework.png)

## Install

We fork from MMDetection. So please follow the installation instructure of [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation). 