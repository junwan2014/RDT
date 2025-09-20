python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage1/diffusion_rl.yaml \
    OUTPUT_DIR output/stage1/rl \
    MODEL.WEIGHTS output/stage1/xe/model_Epoch_00050_Iter_0176999.pth