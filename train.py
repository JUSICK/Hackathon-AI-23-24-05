from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("weights/best(V1.0).pt")

    results = model.train(
        data="data/yolo/dataset.yaml",

        epochs=150,
        patience=25,

        # RTX 3050 6GB — хороший баланс
        imgsz=1024,
        batch=2,
        device=0,

        optimizer="AdamW",

        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # мягкая аугментация для товаров
        degrees=3.0,
        translate=0.05,
        scale=0.3,

        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,

        mosaic=0.5,
        close_mosaic=10,

        mixup=0.0,
        erasing=0.05,

        amp=True,
        workers=4,

        plots=True,
        save=True,
        exist_ok=True,

        name="v3_aug_1024_adamw_rtx3050_6gb"
    )
